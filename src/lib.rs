// Copyright (c) [2024] [Yegor Vaskonyan]
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A simple, id-based, heterogeneous arena allocator.
//!
//! ## Id-based
//!
//! Uses small unique identifiers instead of references to represent allocations.
//! This leverages the type system to statically assign every identifier to the
//! arena it belongs to, ensuring safety without incurring runtime overhead.
//!
//! Accessing individual elements is done through the various
//! arena methods, conceptually similar to indexing a `Vec`.
//!
//! ## Heterogeneous
//!
//! Supports allocating values of all statically sized non-ZST types, which is especially useful
//! in scenarios where you have tree-like data structures with different node types.
//!
//! ## Statically guaranteed safety
//!
//! The implementation leverages the power of the Rust's type
//! system, achieving safety with almost no runtime checks.
//!
//! ## No `Drop`
//!
//! This design, however, has one downside: the arena does not know about individual objects
//! it contains, which makes it impossible to run their destructors on `drop`.
//!
//! ## Examples
//!
//! ```rust
//! use index_arena::{Id, new_arena};
//!
//! struct Even<A> {
//!     next: Option<Id<Odd<A>, A>>,
//! }
//!
//! struct Odd<A> {
//!     next: Option<Id<Even<A>, A>>,
//! }
//!
//! let mut arena = new_arena!();
//!
//! let three = arena.alloc(Odd { next: None });
//! let two = arena.alloc(Even { next: Some(three) });
//! let one = arena.alloc(Odd { next: Some(two) });
//!
//! assert_eq!(&arena[one].next, &Some(two));
//! ```

use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr;
use core::ptr::NonNull;
use std::ops::{Index, IndexMut};

use aligned_vec::{AVec, ConstAlign};
use derive_where::derive_where;

macro_rules! assert_const {
    ($cond:expr, $($arg:tt)+) => {
        if const { !$cond } {
            assert!($cond, $($arg)+);
        }
    };

    ($cond:expr $(,)?) => {
        if const { !$cond } {
            assert!($cond);
        }
    };
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct RawId {
    byte_offset: u32,
}

/// A unique identifier for an object allocated using `Arena`.
///
/// `Id<T, A>` can only be used with the specific arena from which it was created,
/// thanks to the type parameter `A`, which uniquely identifies the arena.
///
/// An `Id<T, A>` guarantees that calling `Arena::get` with it will always yield
/// a reference to the same object (bitwise identical), unless the object is
/// explicitly mutated via a mutable reference obtained from `Arena::get_mut`.
/// The object associated with this `Id` is guaranteed to have the same lifetime
/// as the arena itself, meaning it remains valid as long as the arena exists.
#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Id<T, A> {
    // Invariant: `raw_id.byte_offset` always represents a valid location
    // within the arena holding a value of type `T`, provided `size_of::<T>() > 0`.
    raw_id: RawId,
    _marker: PhantomData<(fn() -> T, A)>,
}

impl<T, A> Id<T, A> {
    #[inline]
    unsafe fn new(byte_offset: usize) -> Id<T, A> {
        let byte_offset = byte_offset.try_into()
            .expect("`byte_offset` must not exceed `u32::MAX`");

        Id { raw_id: RawId { byte_offset }, _marker: PhantomData }
    }

    /// Creates a new `Id<T, S>` from the given `RawId`.
    /// 
    /// # Safety
    /// `raw_id` must have been created from an `Id<T, A>` with the same `T` and `A`.
    #[inline]
    pub unsafe fn from_raw(raw_id: RawId) -> Id<T, A> {
        Id { raw_id, _marker: PhantomData }
    }

    #[inline]
    pub fn as_raw(&self) -> &RawId {
        &self.raw_id
    }

    #[inline]
    pub fn into_raw(self) -> RawId {
        self.raw_id
    }
}

impl<T, A> Id<MaybeUninit<T>, A> {
    /// Converts this `Id<MaybeUninit<T>, A>` to `Id<T, A>`,
    /// assuming the associated value is initialized.
    ///
    /// # Safety
    /// The caller must ensure the value is fully initialized before calling this method.
    #[inline]
    unsafe fn assume_init(self) -> Id<T, A> {
        Id { raw_id: self.raw_id, _marker: PhantomData }
    }
}

/// A simple heterogeneous arena allocator inspired by the `id_arena` crate.
///
/// Unlike the `id_arena` crate, this implementation allows any type to be allocated
/// within the arena and ensures that identifiers are only valid within the arena they
/// were created from. This guarantees safety with minimal runtime overhead.
///
/// However, this approach has a downside: the arena does not track individual elements,
/// effectively providing a form of type erasure. As a result, it is not possible to
/// implement proper dropping of individual elements like in `id_arena::Arena`.
#[derive_where(Debug)]
pub struct Arena<A, const MAX_ALIGN: usize = 128> {
    storage: AVec<MaybeUninit<u8>, ConstAlign<MAX_ALIGN>>,
    _marker: PhantomData<A>,
}

impl<A, const MAX_ALIGN: usize> Arena<A, MAX_ALIGN> {
    /// Creates a new, empty arena.
    ///
    /// # Safety
    /// The caller must ensure that the `A` type parameter is only used for this arena.
    #[inline]
    pub unsafe fn new() -> Arena<A> {
        Arena {
            storage: AVec::new(0),
            _marker: PhantomData,
        }
    }

    /// Returns a shared reference to the arena-allocated object associated with given `Id`.
    #[inline]
    pub fn get<T>(&self, id: Id<T, A>) -> &T {
        assert_const!(size_of::<T>() != 0);

        // SAFETY: The pointer returned by `self.get_ptr` is guaranteed to be non-null, 
        // properly aligned, and point to an initialized value of type `T`. 
        // Additionally, the arena is borrowed as immutable, upholding aliasing rules.
        unsafe { self.get_ptr(id).as_ref() }
    }

    /// Returns a mutable reference to the arena-allocated object associated with given `Id`.
    #[inline]
    pub fn get_mut<T>(&mut self, id: Id<T, A>) -> &mut T {
        assert_const!(size_of::<T>() != 0);

        // SAFETY: The pointer returned by `self.get_ptr` is guaranteed to be non-null, 
        // properly aligned, and point to an initialized value of type `T`. 
        // Additionally, the arena is borrowed as mutable, upholding aliasing rules.
        unsafe { self.get_ptr(id).as_mut() }
    }

    /// Returns a pointer to the object associated with the given id.
    ///
    /// The returned value is always safe to dereference, provided aliasing rules are not violated.
    #[inline]
    fn get_ptr<T>(&self, id: Id<T, A>) -> NonNull<T> {
        assert_const!(size_of::<T>() != 0 && align_of::<T>() <= MAX_ALIGN);

        // SAFETY: `id.raw_id.byte_offset` points to a valid object of type `T`.
        let ptr = unsafe {
            let raw_ptr = self.storage.as_ptr().add(id.raw_id.byte_offset as usize);
            raw_ptr as *const T
        };

        // SAFETY: `ptr` cannot be null.
        unsafe { NonNull::new_unchecked(ptr.cast_mut()) }
    }

    /// Allocates a new value of type `T` in the arena and returns its `Id`.
    #[inline]
    pub fn alloc<T>(&mut self, item: T) -> Id<T, A> {
        // Allocate a new item without initializing it.
        let id = self.alloc_uninit::<T>();

        // SAFETY: `MaybeUninit::as_mut_ptr` always returns a valid pointer for `ptr::write`.
        unsafe { ptr::write(self.get_mut(id).as_mut_ptr(), item); }

        // SAFETY: we have just initialized the memory associated with `id`.
        unsafe { id.assume_init() }
    }

    #[inline]
    pub fn alloc_uninit<T>(&mut self) -> Id<MaybeUninit<T>, A> {
        assert_const!(
            size_of::<T>() != 0 && 
            align_of::<T>() <= MAX_ALIGN && 
            align_of::<T>().is_power_of_two()
        );

        let layout = Layout::new::<T>();

        // Since the backing storage is aligned to `MAX_ALIGN` and
        // `align_of::<T>() <= MAX_ALIGN`, we only need to ensure that
        // the start of the new allocation is aligned to `layout.align()`.
        // SAFETY: `align_of::<T>()` is guaranteed to be a power of two.
        let padding = unsafe { compute_padding(self.storage.len(), layout.align()) };

        // SAFETY: `compute_padding` ensures that `padding < layout.align()`
        // and `Layout` guarantees that both size and alignment do not exceed
        // `isize::MAX`. Therefore, `layout.size() + padding` can be at most
        // `2 * (isize::MAX as usize)`, which is less than `usize::MAX`.
        let padded_size = unsafe { layout.size().unchecked_add(padding) };

        // We will not be touching the first `padding` bytes leaving them 
        // in an initialized state, which is sound for `MaybeUninit<u8>`.
        let byte_offset = self.alloc_raw(padded_size);

        // We have to ensure that the allocation is properly aligned, so
        // we added the padding.
        // SAFETY: `padding < padded_size`, `grow()` didn't panic and length
        // cannot be less than capacity, so this must not overflow.
        let byte_offset = unsafe { byte_offset.unchecked_add(padding) };

        // SAFETY: the memory location at `byte_offset` is properly
        // aligned to hold a value of type `T` and `MaybeUninit`
        // does not require initialization.
        let id: Id<MaybeUninit<T>, A> = unsafe { Id::new(byte_offset) };

        id
    }

    /// Allocates `size_in_bytes` uninitialized bytes in the arena and
    /// returns the byte offset of the beginning of the allocation.
    #[inline]
    fn alloc_raw(&mut self, size_in_bytes: usize) -> usize {
        self.storage.reserve(size_in_bytes);

        let old_len = self.storage.len();

        // SAFETY: `storage.reserve()` didn't panic and length cannot
        // be less than capacity, so this must not overflow.
        let new_len = unsafe { old_len.unchecked_add(size_in_bytes) };

        // SAFETY: we have just reserved `additional` bytes.
        unsafe { self.storage.set_len(new_len); }

        old_len
    }
}

/// Computes the smallest possible number of bytes that must
/// be added to `addr` to align it to `align` bytes.
///
/// The returned value is always within the range `[0, align)`.
///
/// # Safety
/// `align` must be a power of two.
#[inline]
const unsafe fn compute_padding(addr: usize, align: usize) -> usize {
    // This function is essentially a simplified version of `core::ptr::align_offset`.
    // As such, the correctness of this code depends on the correctness of the latter.

    // SAFETY: `align` is a power of two, so it cannot be zero.
    let align_minus_one = unsafe { align.unchecked_sub(1) };

    // Voodoo magic!

    let aligned_addr = addr.wrapping_add(align_minus_one) & 0usize.wrapping_sub(align);
    let byte_offset = aligned_addr.wrapping_sub(addr);

    // From `std::ptr::align_offset`:
    //
    // Masking by `-align` affects only the low bits, and thus cannot reduce
    // the value by more than `align - 1`. Therefore, even though intermediate
    // values might wrap, the `byte_offset` is always within the range `[0, align)`.
    debug_assert!(byte_offset < align);

    // Correctness check.
    debug_assert!((addr + byte_offset) % align == 0);

    byte_offset
}

impl<T, A> Index<Id<T, A>> for Arena<A> {
    type Output = T;

    #[inline]
    fn index(&self, id: Id<T, A>) -> &Self::Output {
        self.get(id)
    }
}

impl<T, A> IndexMut<Id<T, A>> for Arena<A> {
    #[inline]
    fn index_mut(&mut self, id: Id<T, A>) -> &mut Self::Output {
        self.get_mut(id)
    }
}

#[macro_export]
macro_rules! new_arena {
    () => {
        $crate::new_arena!(Default)
    };

    ($name:ident) => {
        {
            struct $name;
            // SAFETY: `$name` is unique for each macro invocation.
            unsafe { $crate::Arena::<$name>::new() }
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn arena_alloc_one_u8() {
        let mut arena = new_arena!();
        let id = arena.alloc(123u8);
        assert_eq!(arena.get(id), &123u8);
    }

    #[test]
    fn arena_alloc_one_u64() {
        let mut arena = new_arena!();
        let id = arena.alloc(u64::MAX);
        assert_eq!(arena.get(id), &u64::MAX);
    }

    #[test]
    fn arena_alloc_mut() {
        let mut arena = new_arena!();

        let id = arena.alloc(456u32);
        assert_eq!(arena.get(id), &456u32);

        *arena.get_mut(id) += 234;
        assert_eq!(*arena.get(id), 456u32 + 234);
    }

    #[test]
    fn arena_alloc_multiple() {
        let mut arena = new_arena!();

        let a_id = arena.alloc(12u16);
        let b_id = arena.alloc("hell");
        let c_id = arena.alloc(131u128);

        assert_eq!(arena.get(b_id), &"hell");
        assert_eq!(arena.get(a_id), &12u16);
        assert_eq!(arena.get(c_id), &131u128);
    }

    #[test]
    fn arena_alloc_multiple_mut() {
        let mut arena = new_arena!();

        let a_id = arena.alloc(12u16);
        let b_id = arena.alloc("hell");
        let c_id = arena.alloc(131u128);

        assert_eq!(arena[b_id], "hell");
        assert_eq!(arena[a_id], 12u16);
        assert_eq!(arena[c_id], 131u128);

        *arena.get_mut(c_id) = 1;
        assert_eq!(arena.get(c_id), &1);

        *arena.get_mut(b_id) = "heaven";
        *arena.get_mut(a_id) *= 3;

        assert_eq!(arena.get(b_id), &"heaven");
        assert_eq!(*arena.get(a_id), 12u16 * 3);
    }
}
