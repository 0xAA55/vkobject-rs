
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::Debug,
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

/// The trait for the mesh to hold buffers
pub trait BufferForDraw<T>: Debug + Clone
where
	T: BufferVecItem {
	/// Must be able to get the `VkBuffer` handle
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Must be able to flush
	fn flush(&mut self, _cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		Ok(())
	}

	/// Convert to `BufferVec<T>`
	fn convert_to_buffer_vec(self) -> BufferVec<T>;

	/// Convert to `BufferWithType<T>`
	fn convert_to_buffer_with_type(self) -> BufferWithType<T>;
}

impl<T> BufferForDraw<T> for BufferVec<T>
where
	T: BufferVecItem {
	fn get_vk_buffer(&self) -> VkBuffer {
		self.get_vk_buffer()
	}

	fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.flush(cmdbuf)
	}

	fn convert_to_buffer_vec(self) -> BufferVec<T> {
		self
	}

	fn convert_to_buffer_with_type(self) -> BufferWithType<T> {
		BufferWithType::new(self.into_inner())
	}
}

impl<T> BufferForDraw<T> for BufferWithType<T>
where
	T: BufferVecItem {
	fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.get_vk_buffer()
	}

	fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.upload_staging_buffer(cmdbuf)?;
		self.discard_staging_buffer();
		Ok(())
	}

	fn convert_to_buffer_vec(self) -> BufferVec<T> {
		let len = self.len();
		unsafe {BufferVec::from_raw_parts(self.into_inner(), len).unwrap()}
	}

	fn convert_to_buffer_with_type(self) -> BufferWithType<T> {
		self
	}
}

#[derive(Debug, Clone)]
pub struct Mesh<BV, V, BE, E, BI, I, BC, C>
where
	BV: BufferForDraw<V>,
	BE: BufferForDraw<E>,
	BI: BufferForDraw<I>,
	BC: BufferForDraw<C>,
	V: BufferVecItem,
	E: BufferVecItem,
	I: BufferVecItem,
	C: BufferVecItem {
	pub primitive_type: VkPrimitiveTopology,
	pub vertices: BV,
	pub indices: Option<BE>,
	pub instances: Option<BI>,
	pub commands: Option<BC>,
	_vertex_type: PhantomData<V>,
	_element_type: PhantomData<E>,
	_instance_type: PhantomData<I>,
	_command_type: PhantomData<C>,
}

/// If a buffer you don't need, use this for your buffer item type
#[derive(Default, Debug, Clone, Copy, Iterable)]
pub struct UnusedBufferItem {}

/// If a buffer you don't need, use this for your buffer type
pub type UnusedBufferType = BufferVec<UnusedBufferItem>;

/// Use this function to create an unused buffer type
pub fn buffer_unused() -> Option<UnusedBufferType> {
	None
}

impl<BV, V, BE, E, BI, I, BC, C> Mesh<BV, V, BE, E, BI, I, BC, C>
where
	BV: BufferForDraw<V>,
	BE: BufferForDraw<E>,
	BI: BufferForDraw<I>,
	BC: BufferForDraw<C>,
	V: BufferVecItem,
	E: BufferVecItem,
	I: BufferVecItem,
	C: BufferVecItem {
	pub fn new(primitive_type: VkPrimitiveTopology, vertices: BV, indices: Option<BE>, instances: Option<BI>, commands: Option<BC>) -> Self {
		Self {
			primitive_type,
			vertices,
			indices,
			instances,
			commands,
			_vertex_type: PhantomData,
			_element_type: PhantomData,
			_instance_type: PhantomData,
			_command_type: PhantomData,
		}
	}

	pub fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.vertices.flush(cmdbuf)?;
		if let Some(ref mut indices) = self.indices {indices.flush(cmdbuf)?;}
		if let Some(ref mut instances) = self.instances {instances.flush(cmdbuf)?;}
		if let Some(ref mut commands) = self.commands {commands.flush(cmdbuf)?;}
		Ok(())
	}
}
