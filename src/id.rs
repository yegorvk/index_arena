use crate::{assert_const, MAX_ALIGN};
use core::{fmt::Debug, hash::Hash, marker::PhantomData, mem::MaybeUninit, slice};
use derive_where::derive_where;

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
/// `Id<T, A>` guarantees that calling `Arena::get` with it will always yield
/// a reference to the same object (bitwise identical), unless the object is
/// explicitly mutated via a mutable reference obtained from `Arena::get_mut`.
/// The object associated with this `Id` is guaranteed to have the same lifetime
/// as the arena itself, meaning it remains valid as long as the arena exists.
#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Id<T: ?Sized + SpecId<A>, A> {
    pub(crate) id: T::Id,
}

impl<T: ?Sized + SpecId<A>, A> Id<T, A> {
    #[inline]
    pub(crate) fn new(spec: T::Id) -> Id<T, A> {
        Id { id: spec }
    }

    #[inline]
    pub(crate) fn get(self, storage_view: &[MaybeUninit<u8>]) -> &T {
        let byte_offset = <T as SpecId<A>>::get_raw_id(self.id).byte_offset as usize;
        let bytes = unsafe { storage_view.get_unchecked(byte_offset..) };
        unsafe { T::get(bytes, self.id) }
    }

    #[inline]
    pub(crate) fn get_mut(self, storage_view: &mut [MaybeUninit<u8>]) -> &mut T {
        let byte_offset = <T as SpecId<A>>::get_raw_id(self.id).byte_offset as usize;
        let bytes = unsafe { storage_view.get_unchecked_mut(byte_offset..) };
        unsafe { T::get_mut(bytes, self.id) }
    }

    #[inline]
    pub fn get_raw_id(&self) -> RawId {
        T::get_raw_id(self.id)
    }
}

impl<T: SpecId<A>, A> From<Id<T, A>> for RawId {
    #[inline]
    fn from(id: Id<T, A>) -> Self {
        id.get_raw_id()
    }
}

/// `Id` specialization for statically sized types.
///
/// All the guarantees `Id` makes also apply for this type.
#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub(crate) struct SizedId<T, A> {
    // Invariant: `byte_offset` always represents a valid location
    // within the arena holding a value of type `T`, provided `size_of::<T>() > 0`.
    byte_offset: u32,
    _marker: PhantomData<(T, A)>,
}

impl<T, A> SizedId<T, A> {
    #[inline]
    pub(crate) unsafe fn new(byte_offset: usize) -> SizedId<T, A> {
        let byte_offset: u32 = byte_offset
            .try_into()
            .expect("`byte_offset` must not exceed `u32::MAX`");

        SizedId {
            byte_offset,
            _marker: PhantomData,
        }
    }
}

impl<T, A> SizedId<MaybeUninit<T>, A> {
    /// Converts this `SizedId<MaybeUninit<T>, A>` to `SizedId<T, A>`,
    /// assuming the associated value is initialized.
    ///
    /// # Safety
    /// The caller must ensure the value is fully initialized before calling this method.
    #[inline]
    pub(crate) unsafe fn assume_init(self) -> SizedId<T, A> {
        SizedId {
            byte_offset: self.byte_offset,
            _marker: PhantomData,
        }
    }
}

/// `Id` specialization for slices.
///
/// All the guarantees `Id` makes also apply for this type.
#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct SliceId<T, A> {
    byte_offset: u32,
    len: u32,
    _marker: PhantomData<(T, A)>,
}

impl<T, A> SliceId<T, A> {
    #[inline]
    pub(crate) unsafe fn new(byte_offset: usize, len: usize) -> SliceId<T, A> {
        let byte_offset: u32 = byte_offset
            .try_into()
            .expect("`byte_offset` must not exceed `u32::MAX`");

        let len: u32 = len.try_into().expect("`len` must not exceed `u32::MAX`");

        SliceId {
            byte_offset,
            len,
            _marker: PhantomData,
        }
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
    pub(crate) unsafe fn assume_init(self) -> SliceId<T, A> {
        SliceId {
            byte_offset: self.byte_offset,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

/// `Id` specialization for string slices.
///
/// The underlying slice always represents a valid UTF-8 encoded string.
/// All the guarantees `Id` makes also apply for this type.
#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub(crate) struct StrId<A>(SliceId<u8, A>);

impl<A> StrId<A> {
    #[inline]
    pub(crate) unsafe fn new(slice_id: SliceId<u8, A>) -> StrId<A> {
        StrId(slice_id)
    }
}

pub(crate) trait SpecId<A> {
    type Id: Debug + Copy + Clone + Eq + PartialEq + Hash;

    /// # Safety
    /// There must be a prefix of `bytes` representing the element referred to by `id`.
    unsafe fn get(bytes: &[MaybeUninit<u8>], id: Self::Id) -> &Self;

    /// # Safety
    /// There must be a prefix of `bytes` representing the element referred to by `id`.
    unsafe fn get_mut(bytes: &mut [MaybeUninit<u8>], id: Self::Id) -> &mut Self;

    fn get_raw_id(id: Self::Id) -> RawId;
}

impl<T, A> SpecId<A> for T {
    type Id = SizedId<T, A>;

    #[inline]
    unsafe fn get(bytes: &[MaybeUninit<u8>], _id: Self::Id) -> &Self {
        assert_const!(size_of::<T>() != 0 && align_of::<T>() <= MAX_ALIGN);
        debug_assert!((bytes.as_ptr() as usize) % align_of::<T>() == 0);
        let ptr: *const T = bytes.as_ptr().cast();
        unsafe { &*ptr }
    }

    #[inline]
    unsafe fn get_mut(bytes: &mut [MaybeUninit<u8>], _id: Self::Id) -> &mut Self {
        assert_const!(size_of::<T>() != 0 && align_of::<T>() <= MAX_ALIGN);
        debug_assert!((bytes.as_ptr() as usize) % align_of::<T>() == 0);
        let ptr: *mut T = bytes.as_mut_ptr().cast();
        unsafe { &mut *ptr }
    }

    #[inline]
    fn get_raw_id(id: Self::Id) -> RawId {
        RawId {
            byte_offset: id.byte_offset,
        }
    }
}

impl<T, A> SpecId<A> for [T] {
    type Id = SliceId<T, A>;

    unsafe fn get(bytes: &[MaybeUninit<u8>], id: Self::Id) -> &Self {
        assert_const!(size_of::<T>() != 0 && align_of::<T>() <= MAX_ALIGN);
        debug_assert!((bytes.as_ptr() as usize) % align_of::<T>() == 0);
        unsafe { slice::from_raw_parts(bytes.as_ptr().cast(), id.len as usize) }
    }

    unsafe fn get_mut(bytes: &mut [MaybeUninit<u8>], id: Self::Id) -> &mut Self {
        assert_const!(size_of::<T>() != 0 && align_of::<T>() <= MAX_ALIGN);
        debug_assert!((bytes.as_ptr() as usize) % align_of::<T>() == 0);
        unsafe { slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), id.len as usize) }
    }

    fn get_raw_id(id: Self::Id) -> RawId {
        RawId {
            byte_offset: id.byte_offset,
        }
    }
}

impl<A> SpecId<A> for str {
    type Id = StrId<A>;

    unsafe fn get(bytes: &[MaybeUninit<u8>], id: Self::Id) -> &Self {
        let bytes = <[u8] as SpecId<A>>::get(bytes, id.0);
        unsafe { str::from_utf8_unchecked(bytes) }
    }

    unsafe fn get_mut(bytes: &mut [MaybeUninit<u8>], id: Self::Id) -> &mut Self {
        let bytes = <[u8] as SpecId<A>>::get_mut(bytes, id.0);
        unsafe { str::from_utf8_unchecked_mut(bytes) }
    }

    fn get_raw_id(id: Self::Id) -> RawId {
        <[u8] as SpecId<A>>::get_raw_id(id.0)
    }
}
