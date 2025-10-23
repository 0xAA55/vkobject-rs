
use std::{
	any::Any,
	collections::hash_map::DefaultHasher,
	env,
	ffi::CString,
	fmt::Debug,
	fs::{read, write},
	hash::{Hash, Hasher},
	io,
	iter::FromIterator,
	mem::{forget, size_of, size_of_val},
	ops::{Deref, DerefMut},
	slice,
	path::PathBuf,
	thread::sleep,
	time::Duration,
	vec::IntoIter,
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
			(self.destroyer)(resource);
		}
	}
}

/// Generate a temp file path for cache, the filename is hashed
pub fn get_hashed_cache_file_path(cache_usage: &str, extension: Option<&str>) -> PathBuf {
	let exe_path = env::current_exe().unwrap_or(PathBuf::from(env!("CARGO_PKG_NAME")));
	let mut hasher = DefaultHasher::new();
	exe_path.to_string_lossy().to_string().hash(&mut hasher);
	cache_usage.hash(&mut hasher);
	let hash = hasher.finish();
	let mut path = std::env::temp_dir();
	path.push(format!("{hash}"));
	path.set_extension(extension.unwrap_or("tmp"));
	path
}

pub fn load_cache<T: Clone + Copy + Sized>(cache_usage: &str, extension: Option<&str>) -> io::Result<Vec<T>> {
	let path = get_hashed_cache_file_path(cache_usage, extension);
	let mut data = read(&path)?;
	let item_size = size_of::<T>();
	let ptr = data.as_mut_ptr();
	let len = data.len() / item_size;
	let capacity = data.capacity() / item_size;
	unsafe {
		forget(data);
		Ok(Vec::from_raw_parts(ptr as *mut T, len, capacity))
	}
}

pub fn save_cache<T: Clone + Copy + Sized>(cache_usage: &str, extension: Option<&str>, data: &[T]) -> io::Result<()> {
	let path = get_hashed_cache_file_path(cache_usage, extension);
	let ptr = data.as_ptr() as *const u8;
	let len = size_of_val(data);
	let data: &[u8] = unsafe {slice::from_raw_parts(ptr, len)};
	write(&path, data)
}

/// A helper trait for `struct_iterable` crate, give a struct that derives `Iterable` the ability to iterate the member's offset and size
pub trait IterableDataAttrib {
	/// A member that allows to get the iterator, this function is to be implemented
	fn iter_members(&self) -> IntoIter<(&'static str, &dyn Any)>;

	/// Get an iterator that could give you the name, offset, and size of the struct members
	fn iter_members_data_attribs(&self) -> IntoIter<(&'static str, usize, usize)> {
		let mut instance_ptr = None;
		let data_attribs: Vec<_> = self.iter_members().map(|(name, value)|{
			let value_ptr = value as *const dyn Any as *const u8;
			if instance_ptr.is_none() {
				instance_ptr = Some(value_ptr)
			}
			let offset = unsafe { value_ptr.offset_from(instance_ptr.unwrap()) } as usize;
			(name, offset, size_of_val(value))
		}).collect();
		data_attribs.into_iter()
	}
}

pub fn format_size(size: u64) -> String {
	let eib = 1024 * 1024 * 1024 * 1024 * 1024;
	let tib = 1024 * 1024 * 1024 * 1024;
	let gib = 1024 * 1024 * 1024;
	let mib = 1024 * 1024;
	let kib = 1024;
	if size >= eib {
		format!("{:.3} EiB", size as f64 / eib as f64)
	} else if size >= tib {
		format!("{:.3} TiB", size as f64 / tib as f64)
	} else if size >= gib {
		format!("{:.3} GiB", size as f64 / gib as f64)
	} else if size >= mib {
		format!("{:.3} MiB", size as f64 / mib as f64)
	} else if size >= kib {
		format!("{:.3} KiB", size as f64 / kib as f64)
	} else {
		format!("{size} B")
	}
}
