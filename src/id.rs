use core::{fmt::Debug, hash::Hash, marker::PhantomData, mem::MaybeUninit, slice};
use derive_where::derive_where;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct RawId {
    byte_offset: u32,
}

mod private {
    pub(crate) trait Sealed {}
}

/// A type that can be stored inside `Arena`.
#[allow(private_bounds)]
pub trait SupportedType: SpecId + private::Sealed {}

impl<T: ?Sized + SpecId> private::Sealed for T {}
impl<T: ?Sized + SpecId> SupportedType for T {}

/// A unique identifier for an object allocated using `Mrena`.
///
/// `Id<T, M>` can only be used with the specific arena from which it was created,
/// thanks to the type parameter `M`, which uniquely identifies the arena.
///
/// `Id<T, M>` guarantees that calling `Arena::get` with it will always yield
/// a reference to the same object (bitwise identical), unless the object is
/// explicitly mutated via a mutable reference obtained from `Arena::get_mut`.
/// The object associated with this `Id` is guaranteed to have the same lifetime
/// as the arena itself, meaning it remains valid as long as the arena exists.
#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash; T::Id<M>)]
#[repr(transparent)]
pub struct Id<T: ?Sized + SupportedType, M> {
    id: T::Id<M>,
}

impl<T: ?Sized + SupportedType, M> Id<T, M> {
    #[inline]
    fn new(id: T::Id<M>) -> Self {
        Id { id }
    }

    /// # Safety
    /// `storage_bytes` must contain a valid value of type `T` at the byte offset
    ///  derived from the id (```<T as SpecId<M>>::get_raw_id(id.id).byte_offset```).
    #[inline]
    pub(crate) unsafe fn get<'a>(&self, storage_bytes: &'a [MaybeUninit<u8>]) -> &'a T {
        let byte_offset = <T as SpecId>::get_raw_id(&self.id).byte_offset as usize;
        let bytes = unsafe { storage_bytes.get_unchecked(byte_offset..) };
        unsafe { T::get(&self.id, bytes) }
    }

    /// # Safety
    /// `storage_bytes` must contain a valid value of type `T` at the byte offset
    ///  derived from the id (```<T as SpecId<M>>::get_raw_id(id.id).byte_offset```).
    #[inline]
    pub(crate) unsafe fn get_mut<'a>(&self, storage_bytes: &'a mut [MaybeUninit<u8>]) -> &'a mut T {
        let byte_offset = <T as SpecId>::get_raw_id(&self.id).byte_offset as usize;
        let bytes = unsafe { storage_bytes.get_unchecked_mut(byte_offset..) };
        unsafe { T::get_mut(&self.id, bytes) }
    }

    #[inline]
    pub fn raw_id(&self) -> RawId {
        T::get_raw_id(&self.id)
    }
}

#[allow(private_bounds)]
impl<T: SpecId<Id<M> = SizedId<T, M>>, M> Id<T, M> {
    #[inline]
    pub(crate) unsafe fn new_sized(byte_offset: usize) -> Self {
        Id::new(unsafe { SizedId::new(byte_offset) })
    }
}

#[allow(private_bounds)]
impl<T, M> Id<[T], M>
where
    [T]: SpecId<Id<M> = SliceId<T, M>>,
{
    #[inline]
    pub(crate) unsafe fn new_slice(byte_offset: usize, len: usize) -> Self {
        Id::new(unsafe { SliceId::new(byte_offset, len) })
    }
}

impl<M> Id<str, M> {
    #[inline]
    pub(crate) unsafe fn new_str(slice_id: Id<[u8], M>) -> Self {
        Id::new(unsafe { StrId::new(slice_id.id) })
    }
}

impl<T, M> Id<MaybeUninit<T>, M> {
    /// Converts `Id<MaybeUninit<T>, M>` to `Id<T, M>`,
    /// assuming the associated value is initialized.
    ///
    /// # Safety
    /// The caller must ensure the value has been fully initialized before
    /// calling this method.
    #[inline]
    pub unsafe fn assume_init(&self) -> Id<T, M> {
        Id::new(unsafe { self.id.assume_init() })
    }
}

impl<T, M> Id<[MaybeUninit<T>], M> {
    /// Converts `Id<[MaybeUninit<T>], M>` to `Id<T, M>`,
    /// assuming the associated value is initialized.
    ///
    /// # Safety
    /// The caller must ensure the value has been fully initialized before
    /// calling this method.
    #[inline]
    pub unsafe fn assume_init(&self) -> Id<[T], M> {
        Id::new(unsafe { self.id.assume_init() })
    }
}

impl<T: SupportedType, M> From<Id<T, M>> for RawId {
    #[inline]
    fn from(id: Id<T, M>) -> Self {
        id.raw_id()
    }
}

/// `Id` specialization for statically sized types.
///
/// All the guarantees `Id` makes also apply for this type.
#[derive_where(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(transparent)]
struct SizedId<T, M> {
    // Invariant: `byte_offset` always represents a valid location
    // within the arena holding a value of type `T`, provided `size_of::<T>() > 0`.
    byte_offset: u32,
    _marker: PhantomData<(T, M)>,
}

impl<T, M> SizedId<T, M> {
    #[inline]
    unsafe fn new(byte_offset: usize) -> SizedId<T, M> {
        let byte_offset: u32 = byte_offset
            .try_into()
            .expect("`byte_offset` must not exceed `u32::MAX`");

        SizedId {
            byte_offset,
            _marker: PhantomData,
        }
    }
}

impl<T, M> SizedId<MaybeUninit<T>, M> {
    /// Converts `SizedId<MaybeUninit<T>, M>` to `SizedId<T, M>`,
    /// assuming the associated value is initialized.
    ///
    /// # Safety
    /// The caller must ensure the value is fully initialized before calling this method.
    #[inline]
    unsafe fn assume_init(self) -> SizedId<T, M> {
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
struct SliceId<T, M> {
    byte_offset: u32,
    len: u32,
    _marker: PhantomData<(T, M)>,
}

impl<T, M> SliceId<T, M> {
    #[inline]
    unsafe fn new(byte_offset: usize, len: usize) -> SliceId<T, M> {
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

impl<T, M> SliceId<MaybeUninit<T>, M> {
    /// Converts this `SliceId<MaybeUninit<T>, M>` to `SliceId<T, M>`,
    /// assuming the associated value is initialized.
    ///
    /// # Safety
    /// The caller must ensure all slice elements are fully
    /// initialized before calling this method.
    #[inline]
    unsafe fn assume_init(self) -> SliceId<T, M> {
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
struct StrId<M>(SliceId<u8, M>);

impl<M> StrId<M> {
    #[inline]
    unsafe fn new(slice_id: SliceId<u8, M>) -> StrId<M> {
        StrId(slice_id)
    }
}

trait SpecId {
    type Id<M>;
    unsafe fn get<'a, M>(id: &Self::Id<M>, bytes: &'a [MaybeUninit<u8>]) -> &'a Self;
    unsafe fn get_mut<'a, M>(id: &Self::Id<M>, bytes: &'a mut [MaybeUninit<u8>]) -> &'a mut Self;
    fn get_raw_id<M>(id: &Self::Id<M>) -> RawId;
}

impl<T> SpecId for T {
    type Id<M> = SizedId<T, M>;

    #[inline]
    unsafe fn get<'a, M>(_id: &Self::Id<M>, bytes: &'a [MaybeUninit<u8>]) -> &'a Self {
        debug_assert!((bytes.as_ptr() as usize) % align_of::<T>() == 0);
        let ptr: *const T = bytes.as_ptr().cast();
        unsafe { &*ptr }
    }

    #[inline]
    unsafe fn get_mut<'a, M>(_id: &Self::Id<M>, bytes: &'a mut [MaybeUninit<u8>]) -> &'a mut Self {
        debug_assert!((bytes.as_ptr() as usize) % align_of::<T>() == 0);
        let ptr: *mut T = bytes.as_mut_ptr().cast();
        unsafe { &mut *ptr }
    }

    #[inline]
    fn get_raw_id<M>(id: &Self::Id<M>) -> RawId {
        RawId {
            byte_offset: id.byte_offset,
        }
    }
}

impl<T> SpecId for [T] {
    type Id<M> = SliceId<T, M>;

    #[inline]
    unsafe fn get<'a, M>(id: &Self::Id<M>, bytes: &'a [MaybeUninit<u8>]) -> &'a Self {
        debug_assert!((bytes.as_ptr() as usize) % align_of::<T>() == 0);
        unsafe { slice::from_raw_parts(bytes.as_ptr().cast(), id.len as usize) }
    }

    #[inline]
    unsafe fn get_mut<'a, M>(id: &Self::Id<M>, bytes: &'a mut [MaybeUninit<u8>]) -> &'a mut Self {
        debug_assert!((bytes.as_ptr() as usize) % align_of::<T>() == 0);
        unsafe { slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), id.len as usize) }
    }

    #[inline]
    fn get_raw_id<M>(id: &Self::Id<M>) -> RawId {
        RawId {
            byte_offset: id.byte_offset,
        }
    }
}

impl SpecId for str {
    type Id<M> = StrId<M>;

    #[inline]
    unsafe fn get<'a, M>(id: &Self::Id<M>, bytes: &'a [MaybeUninit<u8>]) -> &'a Self {
        let bytes = <[u8] as SpecId>::get(&id.0, bytes);
        unsafe { str::from_utf8_unchecked(bytes) }
    }

    #[inline]
    unsafe fn get_mut<'a, M>(id: &Self::Id<M>, bytes: &'a mut [MaybeUninit<u8>]) -> &'a mut Self {
        let bytes = <[u8] as SpecId>::get_mut(&id.0, bytes);
        unsafe { str::from_utf8_unchecked_mut(bytes) }
    }

    #[inline]
    fn get_raw_id<M>(id: &Self::Id<M>) -> RawId {
        <[u8] as SpecId>::get_raw_id(&id.0)
    }
}
