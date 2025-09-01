
use std::{
	ffi::CString,
	fmt::Debug,
	iter::FromIterator,
	ops::{Deref, DerefMut},
	thread::sleep,
	time::Duration,
};
use rand::prelude::*;

/// A structure to hold owned C-type strings for ffi usages.
#[derive(Debug)]
pub struct CStringArray {
	strings: Vec<CString>,
	cstrarr: Vec<*const i8>,
}

impl CStringArray {
	/// Create a new `CStringArray` from an array of `&str` or `String`.
	pub fn new<T>(input: &[T]) -> Self
	where
		T: Clone + Into<Vec<u8>> {
		let strings: Vec<CString> = input.iter().map(|s|CString::new(s.clone()).unwrap()).collect();
		let cstrarr: Vec<*const i8> = strings.iter().map(|s|s.as_ptr()).collect();
		Self {
			strings,
			cstrarr,
		}
	}

	/// Get the number of the strings
	pub fn len(&self) -> usize {
		self.strings.len()
	}

	/// Get is empty
	pub fn is_empty(&self) -> bool {
		self.strings.is_empty()
	}

	/// Get the pointer to the string list.
	pub fn as_ptr(&self) -> *const *const i8 {
		self.cstrarr.as_ptr()
	}
}

impl<'a> FromIterator<&'a String> for CStringArray {
	fn from_iter<T>(input: T) -> Self
	where
		T: IntoIterator<Item = &'a String> {
		let strings: Vec<CString> = input.into_iter().map(|s|CString::new(s.clone()).unwrap()).collect();
		let cstrarr: Vec<*const i8> = strings.iter().map(|s|s.as_ptr()).collect();
		Self {
			strings,
			cstrarr,
		}
	}
}

impl Clone for CStringArray {
	fn clone(&self) -> Self {
		let strings = self.strings.clone();
		let cstrarr: Vec<*const i8> = strings.iter().map(|s|s.as_ptr()).collect();
		Self {
			strings,
			cstrarr,
		}
	}
}

/// Sleep random nanoseconds, the nanosecond value could be bigger than a second.
pub fn random_sleep<R: Rng>(max_nanos: u64, rng: &mut R) {
	let sleep_nanos = rng.random_range(0..max_nanos);
	let secs = sleep_nanos / 1000000000;
	let nanos = (sleep_nanos % 1000000000) as u32;
	sleep(Duration::new(secs, nanos));
}

/// The error returned from the spin function
#[derive(Debug)]
pub enum SpinError<E: Debug> {
	/// The spin function could not acquire the lock
	SpinFail,

	/// The spin function failed for other reasons
	OtherError(E)
}

pub fn spin_work_with_exp_backoff<T, E: Debug, W: FnMut() -> Result<T, SpinError<E>>>(mut spin_func: W, max_sleep_nanos: u64) -> Result<T, E> {
	let mut sleep_nanos = 1000;
	let mut rng = SmallRng::from_os_rng();
	loop {
		match spin_func() {
			Ok(r) => return Ok(r),
			Err(e) => match e {
				SpinError::SpinFail => {}
				SpinError::OtherError(e) => return Err(e),
			}
		}
		random_sleep(sleep_nanos, &mut rng);
		if sleep_nanos < max_sleep_nanos {
			sleep_nanos = (sleep_nanos * 3) >> 1;
		}
	}
}

/// The resource guard ensures no resource is leaking. On `drop()`, the `destroyer` is used to free the `resource`.
#[derive(Debug)]
pub struct ResourceGuard<R, D: Fn(&R)> {
	resource: Option<R>,
	destroyer: D,
}

impl<R, D: Fn(&R)> ResourceGuard<R, D> {
	/// Create the `ResourceGuard`, `resource` is the resource you want to handle, and `destroyer` is the closure you write to free the resource.
	pub fn new(resource: R, destroyer: D) -> Self {
		Self {
			resource: Some(resource),
			destroyer,
		}
	}

	/// Release the resource from this structure, so it won't be freed when the guard is `drop()`ing.
	pub fn release(mut self) -> R {
		self.resource.take().unwrap()
	}
}

impl<R, D: Fn(&R)> Deref for ResourceGuard<R, D> {
	type Target = R;
	fn deref(&self) -> &R {
		self.resource.as_ref().unwrap()
	}
}

impl<R, D: Fn(&R)> DerefMut for ResourceGuard<R, D> {
	fn deref_mut(&mut self) -> &mut R {
		self.resource.as_mut().unwrap()
	}
}

impl<R, D: Fn(&R)> Drop for ResourceGuard<R, D> {
	fn drop(&mut self) {
		if let Some(resource) = &self.resource {
			(self.destroyer)(&resource);
		}
	}
}
