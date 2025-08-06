use crate::{utils::pad_to_align, DEFAULT_STORAGE_ALIGN};
use core::{fmt::Debug, mem::MaybeUninit};

/// A storage backend for `Arena`.
///
/// The storage is conceptually modeled as a growable linear array of bytes that
/// can be relocated to fit more elements (e.g., `Vec<u8>`), later referred to as
/// the storage buffer. However, since we must support storing arbitrary types,
/// it has to preserve alignment of elements across relocations. For that
/// reason, the base address of the buffer must be aligned to `ALIGN` bytes, so
/// that all alignments up to `ALIGN` are preserved.
///
/// # Safety
///
/// The `Storage` trait is unsafe because users can rely on its contracts for
/// soundness. All implementors must uphold these invariants:
///
/// All methods must strictly follow their safety contracts, as the callers
/// can rely on them to guarantee the soundness of their own code.
///
/// ## Empty storage
///
/// Storage is considered **empty** if and only if `view().len() == 0`  
/// (equivalently, `view_mut().len() == 0`). In that case:
///   - `view` and `view_mut` always return empty slices.
///   - If `try_grow(additional_bytes)` returns `Ok(())` with `additional_bytes > 0`,
///     the storage becomes permanently non-empty. Otherwise, it remains empty.
///
/// ## Relocation and alignment
///
/// - Storage may only grow (never shrink), and `try_grow` may relocate the buffer,
///   invalidating any previous pointers.
/// - After any relocation, the buffer’s start address **must** remain aligned to
///   `ALIGN` bytes. This guarantees that elements with alignment ≤ `ALIGN`
///   remain correctly aligned across relocations.
///
/// ## Preservation of contents
///
/// Regardless of relocations, all methods must preserve all previously
/// stored bytes and their order **bit-for-bit**. However, this only
/// applies to the methods themselves (e.g., `view_mut` returns a
/// mutable slice, which is fine).
pub unsafe trait Storage {
    type AllocError;

    /// The alignment of the storage buffer.
    const ALIGN: usize;

    /// Attempts to increase the storage buffer size by `additional_bytes` bytes.
    ///
    /// If a call to this method succeeds, then `view`/`view_mut` will return slices
    /// with lengths exactly `additional_bytes` greater than before the call.
    fn try_grow_by(&mut self, additional_bytes: usize) -> Result<(), Self::AllocError>;

    /// Returns a view of the storage buffer.
    ///
    /// The only difference between this method and `view_mut` is mutability,
    /// which can be relied upon by the caller.
    fn view(&self) -> &[MaybeUninit<u8>];

    /// Returns a mutable view of the storage buffer.
    ///
    /// The only difference between this method and `view` is mutability,
    /// which can be relied upon by the caller.
    fn view_mut(&mut self) -> &mut [MaybeUninit<u8>];
}

#[cfg(feature = "alloc")]
mod alloc {
    use super::*;
    use aligned_vec::{AVec, ConstAlign};
    use core::convert::Infallible;

    pub struct VecStorage<const ALIGN: usize = DEFAULT_STORAGE_ALIGN> {
        bytes: AVec<MaybeUninit<u8>, ConstAlign<ALIGN>>,
    }

    impl VecStorage {
        #[inline]
        pub fn new() -> Self {
            Self::new_align()
        }
    }

    impl<const ALIGN: usize> VecStorage<ALIGN> {
        #[inline]
        pub fn new_align() -> Self {
            Default::default()
        }
    }

    impl<const ALIGN: usize> Default for VecStorage<ALIGN> {
        #[inline]
        fn default() -> Self {
            Self {
                bytes: AVec::new(ALIGN),
            }
        }
    }

    unsafe impl<const ALIGN: usize> Storage for VecStorage<ALIGN> {
        type AllocError = Infallible;
        const ALIGN: usize = ALIGN;

        #[inline]
        fn try_grow_by(&mut self, additional_bytes: usize) -> Result<(), Self::AllocError> {
            self.bytes.reserve(additional_bytes);

            // SAFETY: `Vec` capacity cannot be less than length.
            let new_size = unsafe { self.bytes.len().unchecked_add(additional_bytes) };

            // SAFETY: `MaybeUninit` doesn't require initialization and we have just
            // reserved `additional_bytes` bytes.
            unsafe {
                self.bytes.set_len(new_size);
            }

            Ok(())
        }

        #[inline]
        fn view(&self) -> &[MaybeUninit<u8>] {
            self.bytes.as_slice()
        }

        #[inline]
        fn view_mut(&mut self) -> &mut [MaybeUninit<u8>] {
            self.bytes.as_mut_slice()
        }
    }
}

#[cfg(feature = "alloc")]
pub use alloc::*;

#[derive(Debug)]
pub enum SliceStorageError {
    Overflow,
    OutOfMemory,
}

pub struct SliceStorage<'a, const ALIGN: usize = DEFAULT_STORAGE_ALIGN> {
    bytes: &'a mut [MaybeUninit<u8>],
    size: usize,
}

impl<'a> SliceStorage<'a> {
    #[inline]
    pub fn from_unaligned_bytes(
        bytes: &'a mut [MaybeUninit<u8>],
    ) -> Result<Self, SliceStorageError> {
        Self::from_unaligned_bytes_align(bytes)
    }
}

impl<'a, const ALIGN: usize> SliceStorage<'a, ALIGN> {
    #[inline]
    pub fn from_unaligned_bytes_align(
        bytes: &'a mut [MaybeUninit<u8>],
    ) -> Result<Self, SliceStorageError> {
        let padding = unsafe { pad_to_align(bytes.as_ptr() as usize, ALIGN) };
        let mut storage = SliceStorage { bytes, size: 0 };

        // Ensure we have enough storage to accomodate `padding` bytes.
        storage.try_grow_by(padding)?;

        // SAFETY: `try_grow` has succeeded, so this must be safe.
        storage.bytes = unsafe { storage.bytes.get_unchecked_mut(storage.size..) };

        // Don't forget reset `size`.
        storage.size = 0;

        Ok(storage)
    }
}

unsafe impl<const ALIGN: usize> Storage for SliceStorage<'_, ALIGN> {
    type AllocError = SliceStorageError;
    const ALIGN: usize = ALIGN;

    #[inline]
    fn try_grow_by(&mut self, additional_bytes: usize) -> Result<(), Self::AllocError> {
        let new_size = self
            .size
            .checked_add(additional_bytes)
            .ok_or(SliceStorageError::Overflow)?;

        if new_size >= self.bytes.len() {
            return Err(SliceStorageError::OutOfMemory);
        }

        // We don't have to initialize `MaybeUninit`.
        self.size = new_size;

        Ok(())
    }

    #[inline]
    fn view(&self) -> &[MaybeUninit<u8>] {
        unsafe { self.bytes.get_unchecked(0..self.size) }
    }

    #[inline]
    fn view_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        unsafe { self.bytes.get_unchecked_mut(0..self.size) }
    }
}
