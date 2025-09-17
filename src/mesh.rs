
use crate::prelude::*;
use std::{
	ffi::c_void,
	marker::PhantomData,
	mem::size_of,
};
use struct_iterable::Iterable;

/// A wrapper for `Buffer`
#[derive(Debug, Clone)]
pub struct BufferWithType<T>
where
	T: BufferVecItem {
	/// The buffer
	buffer: Buffer,

	/// The phantom data to hold the type
	_phantom: PhantomData<T>,
}

impl<T> BufferWithType<T>
where
	T: BufferVecItem {
	/// Create the `BufferWithType<T>`
	pub fn new(buffer: Buffer) -> Self {
		Self {
			buffer,
			_phantom: PhantomData,
		}
	}

	/// Create staging buffer for the `BufferWithType<T>`
	pub fn ensure_staging_buffer(&mut self) -> Result<(), VulkanError> {
		self.buffer.ensure_staging_buffer()
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&mut self) {
		self.buffer.discard_staging_buffer();
	}

	/// Get data by an index
	pub fn get_data(&self, index: usize) -> Option<T> {
		if let Some(staging_buffer) = &self.buffer.staging_buffer {
			let mut ret = T::default();
			staging_buffer.get_data(&mut ret as *mut T as *mut c_void, (index * size_of::<T>()) as VkDeviceSize, size_of::<T>()).ok()?;
			Some(ret)
		} else {
			None
		}
	}

	/// Set data
	pub fn set_data(&mut self, index: usize, data: T) -> Result<(), VulkanError> {
		self.buffer.set_staging_data(&data as *const T as *const c_void, (index * size_of::<T>()) as VkDeviceSize, size_of::<T>())
	}

	/// Upload staging buffer data to buffer
	pub fn upload_staging_buffer(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.buffer.upload_staging_buffer(cmdbuf, 0, self.buffer.get_size())
	}

	/// Get the count of the data
	pub fn len(&self) -> usize {
		self.buffer.get_size() as usize / size_of::<T>()
	}

	/// Get if the buffer is empty
	pub fn is_empty(&self) -> bool {
		self.buffer.get_size() == 0
	}

	/// Get the inner buffer
	pub fn into_inner(self) -> Buffer {
		self.buffer
	}
}

	pub primitive_type: VkPrimitiveTopology,
}

#[derive(Default, Debug, Clone, Copy, Iterable)]
pub struct UnusedBufferItem {}

pub type UnusedBufferType = BufferVec<UnusedBufferItem>;

pub fn buffer_unused() -> Option<UnusedBufferType> {
	None
}

where
	V: BufferVecItem,
	E: BufferVecItem,
	I: BufferVecItem,
	C: BufferVecItem {
		Self {
			primitive_type,
			vertices,
			indices,
			instances,
			commands,
		}
	}

	}
}
