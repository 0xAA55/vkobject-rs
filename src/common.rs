
use std::{
	ffi::CString,
	iter::FromIterator,
};

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
