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
//! Accessing individual elements is achieved via various arena methods,
//! conceptually similar to indexing a `Vec`.
//!
//! ## Heterogeneous
//!
//! Supports allocating values of all `Sized` types as well slices and strings.
//! This is particularly useful for managing tree-like data structures
//! with different node types.
//!
//! ## Statically guaranteed safety
//!
//! The implementation leverages the power of the Rust's type
//! system, achieving safety with almost no runtime checks.
//!
//! ## No `Drop`
//!
//! Due to the way this crate works, the arena cannot track individual allocations,
//! so it doesn't drop its elements, which is a necessary trade off.
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

#![no_std]

use core::alloc::Layout;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Index, IndexMut};
use core::ptr;
use utils::{assert_const, pad_to_align};

use crate::utils::MaybeUninitExt;

mod id;
mod storage;
mod utils;

pub use id::{Id, RawId, SupportedType};
pub use storage::{SliceStorage, SliceStorageError, Storage};

#[cfg(feature = "alloc")]
pub use storage::VecStorage;

/// A simple heterogeneous arena allocator inspired by the `id_arena` crate.
///
/// Unlike the `id_arena` crate, this implementation allows any type to be allocated
/// within the arena and ensures that identifiers are only valid for the arena they
/// were created from. This guarantees safety with minimal runtime overhead.
///
/// However, this approach has a downside: the arena does not track individual elements,
/// effectively providing a form of type erasure. As a result, it is not possible to
/// implement proper dropping of individual elements like `id_arena::Arena`.
pub struct Arena<M, S> {
    storage: S,
    _tag: PhantomData<M>,
}

impl<M, S> Arena<M, S> {
    /// Creates a new, empty arena.
    ///
    /// # Safety
    /// The caller must ensure that `M` is only used for this arena.
    #[inline]
    pub unsafe fn with_storage(storage: S) -> Self {
        Arena {
            storage,
            _tag: PhantomData,
        }
    }
}

impl<M, S: Default> Arena<M, S> {
    /// Creates a new, empty arena.
    ///
    /// # Safety
    /// The caller must ensure that `M` is only used for this arena.
    #[inline]
    pub unsafe fn new() -> Self {
        unsafe { Arena::with_storage(S::default()) }
    }
}

impl<M, S: Storage> Arena<M, S> {
    /// Returns a shared reference to the arena-allocated object associated with given `Id`.
    #[inline]
    pub fn get<T: ?Sized + SupportedType>(&self, id: Id<T, M>) -> &T {
        unsafe { id.get(self.storage.view()) }
    }

    /// Returns a mutable reference to the arena-allocated object associated with given `Id`.
    #[inline]
    pub fn get_mut<T: ?Sized + SupportedType>(&mut self, id: Id<T, M>) -> &mut T {
        unsafe { id.get_mut(self.storage.view_mut()) }
    }

    /// Allocates a new value of type `T` in the arena and returns its `Id`.
    #[inline]
    pub fn try_alloc<T>(&mut self, item: T) -> Result<Id<T, M>, S::AllocError> {
        // Allocate a new item without initializing it.
        let uninit_id = self.try_alloc_uninit::<T>()?;

        // SAFETY: `MaybeUninit::as_mut_ptr` always returns a valid pointer for `ptr::write`.
        unsafe {
            ptr::write(self.get_mut(uninit_id).as_mut_ptr(), item);
        }

        // SAFETY: we have just initialized the memory associated with `id`.
        Ok(unsafe { uninit_id.assume_init() })
    }

    #[inline]
    pub fn try_alloc_slice<T: Clone>(&mut self, slice: &[T]) -> Result<Id<[T], M>, S::AllocError> {
        let uninit_id = self.try_alloc_slice_uninit(slice.len())?;
        <MaybeUninit<T> as MaybeUninitExt<T>>::clone_from_slice(self.get_mut(uninit_id), slice);
        Ok(unsafe { uninit_id.assume_init() })
    }

    #[inline]
    pub fn try_alloc_str(&mut self, str: &str) -> Result<Id<str, M>, S::AllocError> {
        let slice_id = self.try_alloc_slice(str.as_bytes())?;
        Ok(unsafe { Id::new_str(slice_id) })
    }

    #[inline]
    fn try_alloc_uninit<T>(&mut self) -> Result<Id<MaybeUninit<T>, M>, S::AllocError> {
        assert_const!(align_of::<T>() <= S::ALIGN);

        if const { size_of::<T>() == 0 } {
            // SAFETY: `byte_offset` can be any value for ZSTs.
            return unsafe { Ok(Id::new_sized(0)) };
        }

        // SAFETY: `align_of::<T>` cannot exceed `S::ALIGN`.
        let byte_offset = unsafe { self.try_alloc_layout(Layout::new::<T>())? };

        // SAFETY: the memory location at `byte_offset` is properly
        // aligned to hold a value of type `T` and `MaybeUninit`
        // does not require initialization.
        Ok(unsafe { Id::new_sized(byte_offset) })
    }

    #[inline]
    fn try_alloc_slice_uninit<T>(
        &mut self,
        len: usize,
    ) -> Result<Id<[MaybeUninit<T>], M>, S::AllocError> {
        assert_const!(align_of::<T>() <= S::ALIGN);

        if const { size_of::<T>() == 0 } {
            // SAFETY: `byte_offset` can be any value for ZSTs.
            return unsafe { Ok(Id::new_slice(0, len)) };
        }

        let layout = Layout::array::<T>(len).unwrap();

        // SAFETY: the alignment of an array is the same as the alignment of
        // its elements and `align_of::<T>` cannot exceed `S::ALIGN`.
        let byte_offset = unsafe { self.try_alloc_layout(layout)? };

        // SAFETY: the memory location at `byte_offset` is properly
        // aligned to hold a value of type `[T; len]` and `MaybeUninit`
        // does not require initialization.
        Ok(unsafe { Id::new_slice(byte_offset, len) })
    }

    /// Allocates uninitialized memory suitable to hold a value with the given layout
    /// and returns the index of the beginning of the allocation.
    ///
    /// # Safety
    /// `layout.size()` must not be zero and `layout.align()` must not exceed `S::ALIGN`.
    #[inline]
    unsafe fn try_alloc_layout(&mut self, layout: Layout) -> Result<usize, S::AllocError> {
        let old_size = self.storage.view().len();

        // Since the backing storage is aligned to `S::ALIGN` and
        // `layout.align() <= S::ALIGN`, we only need to ensure that
        // the start of the new allocation is aligned to `layout.align()`.
        // SAFETY: `layout.align()` is guaranteed to be a power of two.
        let padding = unsafe { pad_to_align(old_size, layout.align()) };

        // SAFETY: `compute_padding` ensures that `padding < layout.align()`
        // and `Layout` guarantees that both size and alignment do not exceed
        // `isize::MAX`. Therefore, `layout.size() + padding` can be at most
        // `2 * (isize::MAX as usize)`, which is less than `usize::MAX`.
        let padded_size = unsafe { layout.size().unchecked_add(padding) };

        self.storage.try_grow_by(padded_size)?;
        Ok(unsafe { old_size.unchecked_add(padding) })
    }
}

impl<M, S> Arena<M, S>
where
    S: Storage,
    S::AllocError: Debug,
{
    /// Allocates a new value of type `T` in the arena and returns its `Id`.
    #[inline]
    pub fn alloc<T>(&mut self, item: T) -> Id<T, M> {
        self.try_alloc(item).unwrap()
    }

    #[inline]
    pub fn alloc_slice<T: Clone>(&mut self, slice: &[T]) -> Id<[T], M> {
        self.try_alloc_slice(slice).unwrap()
    }

    #[inline]
    pub fn alloc_str(&mut self, str: &str) -> Id<str, M> {
        self.try_alloc_str(str).unwrap()
    }
}

impl<T, M, S> Index<Id<T, M>> for Arena<M, S>
where
    T: ?Sized + SupportedType,
    S: Storage,
{
    type Output = T;

    #[inline]
    fn index(&self, id: Id<T, M>) -> &Self::Output {
        self.get(id)
    }
}

impl<T, M, S> IndexMut<Id<T, M>> for Arena<M, S>
where
    T: ?Sized + SupportedType,
    S: Storage,
{
    #[inline]
    fn index_mut(&mut self, id: Id<T, M>) -> &mut Self::Output {
        self.get_mut(id)
    }
}

#[doc(hidden)]
pub const DEFAULT_STORAGE_ALIGN: usize = 128;

#[macro_export]
macro_rules! new_arena {
    () => {{
        const ALIGN: usize = $crate::DEFAULT_STORAGE_ALIGN;
        let storage = $crate::VecStorage::<ALIGN>::new();
        $crate::new_arena!(storage = storage)
    }};

    (storage = $storage:expr) => {{
        struct M;
        let storage = $storage;
        unsafe { $crate::Arena::<M, _>::with_storage(storage) }
    }};
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::storage::SliceStorage;

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

    #[test]
    fn arena_alloc_str() {
        let mut arena = new_arena!();
        let hello = arena.alloc_str("Hello!");
        assert_eq!(&arena[hello], "Hello!");
    }

    #[test]
    fn arena_alloc_slice_storage() {
        let mut buf = [MaybeUninit::uninit(); 1024];
        let storage = SliceStorage::from_unaligned_bytes(&mut buf).unwrap();
        let mut arena = new_arena!(storage = storage);
        let counter = arena.alloc(123);
        let fruits = arena.alloc_slice(&["banana", "orange", "apple"]);
        assert_eq!(arena[counter], 123);
        assert_eq!(arena[fruits], ["banana", "orange", "apple"]);
        arena[counter] = 43;
        assert_eq!(arena[counter], 43);
        arena[fruits][0] = "pineapple";
        assert_eq!(arena[fruits][0], "pineapple");
        let hello = arena.alloc_str("Hello!");
        assert_eq!(&arena[hello], "Hello!");
    }

    #[test]
    #[should_panic]
    fn arena_alloc_slice_storage_out_of_memory() {
        #[repr(align(128))]
        struct AlignedSlice([MaybeUninit<u8>; 128]);
        assert!(align_of::<AlignedSlice>() == 128);
        let mut buf: AlignedSlice = AlignedSlice([MaybeUninit::uninit(); 128]);
        let storage = SliceStorage::from_unaligned_bytes(&mut buf.0).unwrap();
        let mut arena = new_arena!(storage = storage);
        arena.alloc_slice(&[0; 256]);
    }

    #[test]
    fn arena_alloc_slice_storage_out_of_memory_fallible() {
        #[repr(align(128))]
        struct AlignedSlice([MaybeUninit<u8>; 128]);
        assert!(align_of::<AlignedSlice>() == 128);
        let mut buf: AlignedSlice = AlignedSlice([MaybeUninit::uninit(); 128]);
        let storage = SliceStorage::from_unaligned_bytes(&mut buf.0).unwrap();
        let mut arena = new_arena!(storage = storage);
        assert!(arena.try_alloc_slice(&[0; 256]).is_err());
    }

    #[test]
    fn arena_alloc_zst() {
        let mut arena = new_arena!();
        let a = arena.alloc(());
        let _ = arena.get(a);
        let b = arena.alloc_slice(&[(); 1024]);
        let _ = arena.get(b);
    }
}
