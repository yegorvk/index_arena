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

#![allow(private_bounds)]

use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Index, IndexMut};
use std::slice::{from_raw_parts, from_raw_parts_mut};

use aligned_vec::{AVec, ConstAlign};
use derive_where::derive_where;

use crate::utils::MaybeUninitExt;

mod utils;

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

const MAX_ALIGN: usize = 128;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct RawId {
    byte_offset: u32,
}

/// `Id` specialization for statically sized types.
///
/// All the guarantees `Id` makes also apply for this type.
#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
struct SizedId<T, A> {
    // Invariant: `byte_offset` always represents a valid location
    // within the arena holding a value of type `T`, provided `size_of::<T>() > 0`.
    byte_offset: u32,
    _marker: PhantomData<(T, A)>,
}

impl<T, A> SizedId<T, A> {
    #[inline]
    unsafe fn new(byte_offset: usize) -> SizedId<T, A> {
        let byte_offset: u32 = byte_offset.try_into()
            .expect("`byte_offset` must not exceed `u32::MAX`");

        SizedId { byte_offset, _marker: PhantomData }
    }
}

impl<T, A> SizedId<MaybeUninit<T>, A> {
    /// Converts this `SizedId<MaybeUninit<T>, A>` to `SizedId<T, A>`,
    /// assuming the associated value is initialized.
    ///
    /// # Safety
    /// The caller must ensure the value is fully initialized before calling this method.
    #[inline]
    unsafe fn assume_init(self) -> SizedId<T, A> {
        SizedId { byte_offset: self.byte_offset, _marker: PhantomData }
    }
}

#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct SliceId<T, A> {
    byte_offset: u32,
    len: u32,
    _marker: PhantomData<(T, A)>,
}

impl<T, A> SliceId<T, A> {
    #[inline]
    unsafe fn new(byte_offset: usize, len: usize) -> SliceId<T, A> {
        let byte_offset: u32 = byte_offset.try_into()
            .expect("`byte_offset` must not exceed `u32::MAX`");

        let len: u32 = len.try_into()
            .expect("`len` must not exceed `u32::MAX`");

        SliceId { byte_offset, len, _marker: PhantomData }
    }
}

impl<T, A> SliceId<MaybeUninit<T>, A> {
    /// Converts this `SliceId<MaybeUninit<T>, A>` to `SliceId<T, A>`,
    /// assuming the associated value is initialized.
    ///
    /// # Safety
    /// The caller must ensure all slice elements are fully
    /// initialized before calling this method.
    #[inline]
    unsafe fn assume_init(self) -> SliceId<T, A> {
        SliceId { byte_offset: self.byte_offset, len: self.len, _marker: PhantomData }
    }
}

/// Describes an `Id` specialization.
///
/// # Safety
/// Implementors must uphold all the guarantees `Id` makes.
unsafe trait SpecId<A> {
    type Id: Debug + Copy + Clone + Eq + PartialEq + Hash;

    fn get(arena: &Arena<A>, id: Self::Id) -> &Self;

    fn get_mut(arena: &mut Arena<A>, id: Self::Id) -> &mut Self;

    fn get_raw_id(id: Self::Id) -> RawId;
}

unsafe impl<T, A> SpecId<A> for T {
    type Id = SizedId<T, A>;

    #[inline]
    fn get(arena: &Arena<A>, id: Self::Id) -> &Self {
        assert_const!(size_of::<T>() != 0 && align_of::<T>() <= MAX_ALIGN);

        let byte_offset = id.byte_offset as usize;

        let ptr = unsafe {
            let raw_ptr = arena.storage.as_ptr().add(byte_offset);
            raw_ptr.cast()
        };

        unsafe { &*ptr }
    }

    #[inline]
    fn get_mut(arena: &mut Arena<A>, id: Self::Id) -> &mut Self {
        assert_const!(size_of::<T>() != 0 && align_of::<T>() <= MAX_ALIGN);

        let byte_offset = id.byte_offset as usize;

        let ptr = unsafe {
            let raw_ptr = arena.storage.as_mut_ptr().add(byte_offset);
            raw_ptr.cast()
        };

        unsafe { &mut *ptr }
    }

    #[inline]
    fn get_raw_id(id: Self::Id) -> RawId {
        RawId { byte_offset: id.byte_offset }
    }
}

unsafe impl<T, A> SpecId<A> for [T] {
    type Id = SliceId<T, A>;

    fn get(arena: &Arena<A>, id: Self::Id) -> &Self {
        let byte_offset = id.byte_offset as usize;
        let len = id.len as usize;

        let ptr = unsafe {
            let raw_ptr = arena.storage.as_ptr().add(byte_offset);
            raw_ptr.cast()
        };

        unsafe { from_raw_parts(ptr, len) }
    }

    fn get_mut(arena: &mut Arena<A>, id: Self::Id) -> &mut Self {
        let byte_offset = id.byte_offset as usize;
        let len = id.len as usize;

        let ptr = unsafe {
            let raw_ptr = arena.storage.as_mut_ptr().add(byte_offset);
            raw_ptr.cast()
        };

        unsafe { from_raw_parts_mut(ptr, len) }
    }

    fn get_raw_id(id: Self::Id) -> RawId {
        RawId { byte_offset: id.byte_offset }
    }
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
pub struct Id<T: ?Sized + SpecId<A>, A> {
    spec: T::Id,
}

impl<T: ?Sized + SpecId<A>, A> Id<T, A> {
    #[inline]
    fn new(spec: T::Id) -> Id<T, A> {
        Id { spec }
    }

    #[inline]
    fn get(self, arena: &Arena<A>) -> &T {
        T::get(arena, self.spec)
    }

    #[inline]
    fn get_mut(self, arena: &mut Arena<A>) -> &mut T {
        T::get_mut(arena, self.spec)
    }

    #[inline]
    pub fn to_raw_id(&self) -> RawId {
        T::get_raw_id(self.spec)
    }
}

impl<T: SpecId<A>, A> From<Id<T, A>> for RawId {
    #[inline]
    fn from(id: Id<T, A>) -> Self {
        id.to_raw_id()
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
pub struct Arena<A> {
    storage: AVec<MaybeUninit<u8>, ConstAlign<MAX_ALIGN>>,
    _marker: PhantomData<A>,
}

impl<A> Arena<A> {
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
    pub fn get<T: ?Sized + SpecId<A>>(&self, id: Id<T, A>) -> &T {
        id.get(self)
    }

    /// Returns a mutable reference to the arena-allocated object associated with given `Id`.
    #[inline]
    pub fn get_mut<T: ?Sized + SpecId<A>>(&mut self, id: Id<T, A>) -> &mut T {
        id.get_mut(self)
    }

    /// Allocates a new value of type `T` in the arena and returns its `Id`.
    #[inline]
    pub fn alloc<T>(&mut self, item: T) -> Id<T, A> {
        // Allocate a new item without initializing it.
        let id = self.alloc_uninit::<T>();

        // SAFETY: `MaybeUninit::as_mut_ptr` always returns a valid pointer for `ptr::write`.
        unsafe { ptr::write(self.get_mut(id).as_mut_ptr(), item); }

        // SAFETY: we have just initialized the memory associated with `id`.
        unsafe { Id::new(id.spec.assume_init()) }
    }

    #[inline]
    pub fn alloc_slice<T: Clone>(&mut self, slice: &[T]) -> Id<[T], A> {
        let id = self.alloc_slice_uninit(slice.len());
        <MaybeUninit<T> as MaybeUninitExt<T>>::clone_from_slice(self.get_mut(id), slice);
        unsafe { Id::new(id.spec.assume_init()) }
    }

    #[inline]
    fn alloc_uninit<T>(&mut self) -> Id<MaybeUninit<T>, A> {
        assert_const!(align_of::<T>() <= MAX_ALIGN && size_of::<T>() != 0);

        // SAFETY: `align_of::<T>` cannot exceed `MAX_ALIGN`.
        let byte_offset = unsafe { self.alloc_layout(Layout::new::<T>()) };

        // SAFETY: the memory location at `byte_offset` is properly
        // aligned to hold a value of type `T` and `MaybeUninit`
        // does not require initialization.
        unsafe { Id::new(SizedId::new(byte_offset)) }
    }

    #[inline]
    fn alloc_slice_uninit<T>(&mut self, len: usize) -> Id<[MaybeUninit<T>], A> {
        assert_const!(align_of::<T>() <= MAX_ALIGN && size_of::<T>() != 0);

        let layout = Layout::array::<T>(len).unwrap();

        // SAFETY: the alignment of an array is the same as the alignment of
        // its elements and `align_of::<T>` cannot exceed `MAX_ALIGN`.
        let byte_offset = unsafe { self.alloc_layout(layout) };

        // SAFETY: the memory location at `byte_offset` is properly
        // aligned to hold a value of type `[T; len]` and `MaybeUninit`
        // does not require initialization.
        unsafe { Id::new(SliceId::new(byte_offset, len)) }
    }

    /// Allocates uninitialized memory suitable to hold a value with the given layout
    /// and returns the index of the beginning of the allocation.
    ///
    /// # Safety
    /// `layout.size()` must not be zero and `layout.align()` must not exceed `MAX_ALIGN`.
    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> usize {
        // Since the backing storage is aligned to `MAX_ALIGN` and
        // `layout.align() <= MAX_ALIGN`, we only need to ensure that
        // the start of the new allocation is aligned to `layout.align()`.
        // SAFETY: `layout.align()` is guaranteed to be a power of two.
        let padding = unsafe { compute_padding(self.storage.len(), layout.align()) };

        // SAFETY: `compute_padding` ensures that `padding < layout.align()`
        // and `Layout` guarantees that both size and alignment do not exceed
        // `isize::MAX`. Therefore, `layout.size() + padding` can be at most
        // `2 * (isize::MAX as usize)`, which is less than `usize::MAX`.
        let padded_size = unsafe { layout.size().unchecked_add(padding) };

        let unaligned_byte_offset = self.alloc_raw(padded_size);

        // SAFETY: `padding < padded_size`, `grow()` didn't panic and length
        // cannot be less than capacity, so this must not overflow.
        unsafe { unaligned_byte_offset.unchecked_add(padding) }
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

impl<T: ?Sized + SpecId<A>, A> Index<Id<T, A>> for Arena<A> {
    type Output = T;

    #[inline]
    fn index(&self, id: Id<T, A>) -> &Self::Output {
        self.get(id)
    }
}

impl<T: ?Sized + SpecId<A>, A> IndexMut<Id<T, A>> for Arena<A> {
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
    fn arena_alloc_multiple_sized() {
        let mut arena = new_arena!();

        let a_id = arena.alloc(12u16);
        let b_id = arena.alloc("hell");
        let c_id = arena.alloc(131u128);

        assert_eq!(arena.get(b_id), &"hell");
        assert_eq!(arena.get(a_id), &12u16);
        assert_eq!(arena.get(c_id), &131u128);
    }

    #[test]
    fn arena_alloc_multiple_sized_mut() {
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
    
    #[test]
    fn arena_alloc_slice() {
        let mut arena = new_arena!();
        let fruits = ["banana", "orange", "apple"];
        let id = arena.alloc_slice(&fruits);
        assert_eq!(arena[id][1], "orange");
    }

    #[test]
    fn arena_alloc_slice_mut() {
        let mut arena = new_arena!();
        let numbers = [12i64, -451i64, 0i64];
        let id = arena.alloc_slice(&numbers);
        assert_eq!(arena[id][1], -451i64);
        arena[id][1] = 3i64;
        assert_eq!(arena[id][1], 3i64);
    }
    
    #[test]
    fn arena_alloc_multiple() {
        let mut arena = new_arena!();
        let counter = arena.alloc(123);
        let fruits = arena.alloc_slice(&["banana", "orange", "apple"]);
        assert_eq!(arena[counter], 123);
        assert_eq!(arena[fruits], ["banana", "orange", "apple"]);
    }

    #[test]
    fn arena_alloc_multiple_mut() {
        let mut arena = new_arena!();
        let counter = arena.alloc(123);
        let fruits = arena.alloc_slice(&["banana", "orange", "apple"]);
        assert_eq!(arena[counter], 123);
        assert_eq!(arena[fruits], ["banana", "orange", "apple"]);
        arena[counter] = 43;
        assert_eq!(arena[counter], 43);
        arena[fruits][0] = "pineapple";
        assert_eq!(arena[fruits][0], "pineapple");
    }
}