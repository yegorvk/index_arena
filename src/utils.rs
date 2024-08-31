use std::mem;
use std::mem::MaybeUninit;
use std::ptr::drop_in_place;

struct Guard<'a, T> {
    slice: &'a mut [MaybeUninit<T>],
    initialized: usize,
}

impl<'a, T> Drop for Guard<'a, T> {
    fn drop(&mut self) {
        let initialized_part = &mut self.slice[..self.initialized];
        // SAFETY: this raw sub-slice will contain only initialized objects.
        unsafe {
            drop_in_place(MaybeUninitExt::slice_assume_init_mut(initialized_part));
        }
    }
}

pub trait MaybeUninitExt<T> {
    /// Assuming all the elements are initialized, get a mutable slice to them.
    ///
    /// # Safety
    ///
    /// It is up to the caller to guarantee that the `MaybeUninit<T>` elements
    /// really are in an initialized state.
    /// Calling this when the content is not yet fully initialized causes undefined behavior.
    ///
    /// See [`assume_init_mut`] for more details and examples.
    ///
    /// [`assume_init_mut`]: MaybeUninit::assume_init_mut
    unsafe fn slice_assume_init_mut(slice: &mut [Self]) -> &mut [T]
    where
        Self: Sized;

    /// Clones the elements from `src` to `this`, returning a mutable reference to the now initialized contents of `this`.
    /// Any already initialized elements will not be dropped.
    ///
    /// If `T` implements `Copy`, use [`copy_from_slice`]
    ///
    /// This is similar to [`slice::clone_from_slice`] but does not drop existing elements.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths, or if the implementation of `Clone` panics.
    ///
    /// If there is a panic, the already cloned elements will be dropped.
    /// [`copy_from_slice`]: MaybeUninit::copy_from_slice
    fn clone_from_slice<'a>(this: &'a mut [MaybeUninit<T>], src: &[T]) -> &'a mut [T]
    where
        T: Clone;
}

impl<T> MaybeUninitExt<T> for MaybeUninit<T> {
    #[inline(always)]
    unsafe fn slice_assume_init_mut(slice: &mut [Self]) -> &mut [T] {
        // SAFETY: similar to safety notes for `slice_get_ref`, but we have a
        // mutable reference which is also guaranteed to be valid for writes.
        unsafe { &mut *(slice as *mut [Self] as *mut [T]) }
    }

    fn clone_from_slice<'a>(this: &'a mut [MaybeUninit<T>], src: &[T]) -> &'a mut [T]
    where
        T: Clone,
    {
        // unlike copy_from_slice this does not call clone_from_slice on the slice
        // this is because `MaybeUninit<T: Clone>` does not implement Clone.

        assert_eq!(this.len(), src.len(), "destination and source slices have different lengths");
        // NOTE: We need to explicitly slice them to the same length
        // for bounds checking to be elided, and the optimizer will
        // generate memcpy for simple cases (for example T = u8).
        let len = this.len();
        let src = &src[..len];

        // guard is needed b/c panic might happen during a clone
        let mut guard = Guard { slice: this, initialized: 0 };

        for i in 0..len {
            guard.slice[i].write(src[i].clone());
            guard.initialized += 1;
        }

        mem::forget(guard);

        // SAFETY: Valid elements have just been written into `this` so it is initialized
        unsafe { MaybeUninitExt::slice_assume_init_mut(this) }
    }
}