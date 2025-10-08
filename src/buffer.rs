
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	marker::PhantomData,
	mem::size_of,
	sync::Arc,
};
use struct_iterable::Iterable;

/// The Vulkan buffer object, same as the OpenGL buffer object, could be used to store vertices, elements(indices), and the other data.
pub struct Buffer {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The buffer
	pub buffer: Arc<VulkanBuffer>,

	/// The device memory
	pub memory: Arc<VulkanMemory>,

	/// The usage of the buffer, not including `VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT` and `VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT`
	pub(crate) usage: VkBufferUsageFlags,

	/// The staging buffer
	pub staging_buffer: Option<StagingBuffer>,
}

impl Buffer {
	/// Create a new buffer
	/// * If `data` is `None`, `cmdbuf` could be `null()` because no `vkCmdCopyBuffer()` will be issued.
	pub fn new(device: Arc<VulkanDevice>, size: VkDeviceSize, data: Option<*const c_void>, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let buffer = Arc::new(VulkanBuffer::new(device.clone(), size, usage | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT as VkBufferUsageFlags)?);
		let memory = Arc::new(VulkanMemory::new(device.clone(), &buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as VkMemoryPropertyFlags)?);
		memory.bind_vk_buffer(buffer.get_vk_buffer())?;
		let mut ret = Self {
			device,
			memory,
			buffer,
			usage,
			staging_buffer: None,
		};
		if let Some(data) = data {
			unsafe {ret.set_staging_data(data, 0, size as usize)?};
		}
		Ok(ret)
	}

	/// Get the `VkBuffer`
	pub(crate) fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.get_vk_buffer()
	}

	/// Create the staging buffer if not exist
	pub fn ensure_staging_buffer(&mut self) -> Result<(), VulkanError> {
		if self.staging_buffer.is_none() {
			self.staging_buffer = Some(StagingBuffer::new(self.device.clone(), self.buffer.get_size())?);
		}
		Ok(())
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&mut self) {
		self.staging_buffer = None;
	}

	/// Get the staging buffer
	pub fn get_staging_buffer(&self) -> Option<&StagingBuffer> {
		self.staging_buffer.as_ref()
	}

	/// Get the usage
	pub fn get_usage(&self) -> VkBufferUsageFlags {
		self.usage
	}

	/// Get the size
	pub fn get_size(&self) -> VkDeviceSize {
		self.buffer.get_size()
	}

	/// Get the address of the staging buffer memory data
	pub fn get_staging_buffer_address(&mut self) -> Result<*mut c_void, VulkanError> {
		self.ensure_staging_buffer()?;
		Ok(self.staging_buffer.as_ref().unwrap().get_address())
	}

	/// Update new data to the buffer
	///
	/// # Safety
	///
	/// You must provide a valid pointer `data`, otherwise the behavior of this function is undefined.
	pub unsafe fn set_staging_data(&mut self, data: *const c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		self.ensure_staging_buffer()?;
		self.staging_buffer.as_mut().unwrap().set_data(data, offset, size)?;
		Ok(())
	}

	/// Retrieve the data from the staging buffer
	///
	/// # Safety
	///
	/// You must provide a valid pointer `data`, otherwise the behavior of this function is undefined.
	pub unsafe fn get_staging_data(&mut self, data: *mut c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		if let Some(ref mut staging_buffer) = self.staging_buffer {
			staging_buffer.get_data(data, offset, size)
		} else {
			Err(VulkanError::NoStagingBuffer)
		}
	}

	/// Upload the data from the staging buffer
	pub fn upload_staging_buffer(&self, cmdbuf: VkCommandBuffer, offset: VkDeviceSize, size: VkDeviceSize) -> Result<(), VulkanError> {
		if let Some(ref staging_buffer) = self.staging_buffer {
			let copy_region = VkBufferCopy {
				srcOffset: offset,
				dstOffset: offset,
				size: size as VkDeviceSize,
			};
			self.device.vkcore.vkCmdCopyBuffer(cmdbuf, staging_buffer.get_vk_buffer(), self.buffer.get_vk_buffer(), 1, &copy_region)?;
			Ok(())
		} else {
			Err(VulkanError::NoStagingBuffer)
		}
	}

	/// Upload the data from the staging buffer
	pub fn upload_staging_buffer_multi(&self, cmdbuf: VkCommandBuffer, regions: &[BufferRegion]) -> Result<(), VulkanError> {
		if let Some(ref staging_buffer) = self.staging_buffer {
			let copy_regions: Vec<VkBufferCopy> = regions.iter().map(|r|VkBufferCopy {
				srcOffset: r.offset,
				dstOffset: r.offset,
				size: r.size as VkDeviceSize,
			}).collect();
			self.device.vkcore.vkCmdCopyBuffer(cmdbuf, staging_buffer.get_vk_buffer(), self.buffer.get_vk_buffer(), copy_regions.len() as u32, copy_regions.as_ptr())?;
			Ok(())
		} else {
			Err(VulkanError::NoStagingBuffer)
		}
	}

	/// Download the data to the staging buffer
	pub fn download_staging_buffer(&mut self, cmdbuf: VkCommandBuffer, offset: VkDeviceSize, size: VkDeviceSize) -> Result<(), VulkanError> {
		self.ensure_staging_buffer()?;
		let copy_region = VkBufferCopy {
			srcOffset: offset,
			dstOffset: offset,
			size: size as VkDeviceSize,
		};
		self.device.vkcore.vkCmdCopyBuffer(cmdbuf, self.buffer.get_vk_buffer(), self.staging_buffer.as_ref().unwrap().get_vk_buffer(), 1, &copy_region)?;
		Ok(())
	}

	/// Download the data to the staging buffer
	pub fn download_staging_buffer_multi(&mut self, cmdbuf: VkCommandBuffer, regions: &[BufferRegion]) -> Result<(), VulkanError> {
		self.ensure_staging_buffer()?;
		let copy_regions: Vec<VkBufferCopy> = regions.iter().map(|r|VkBufferCopy {
			srcOffset: r.offset,
			dstOffset: r.offset,
			size: r.size as VkDeviceSize,
		}).collect();
		self.device.vkcore.vkCmdCopyBuffer(cmdbuf, self.buffer.get_vk_buffer(), self.staging_buffer.as_ref().unwrap().get_vk_buffer(), copy_regions.len() as u32, copy_regions.as_ptr())?;
		Ok(())
	}

	/// Map the staging buffer
	pub fn map_staging_buffer<'a>(&'a mut self, offset: VkDeviceSize, size: usize) -> Result<MappedMemory<'a>, VulkanError> {
		self.ensure_staging_buffer()?;
		let staging_buffer = self.staging_buffer.as_mut().unwrap();
		staging_buffer.memory.map(offset, size)
	}

	/// Create a buffer view
	pub fn create_buffer_view(&self, format: VkFormat) -> Result<VulkanBufferView, VulkanError> {
		VulkanBufferView::new(self.buffer.clone(), format)
	}

	/// Create a buffer view
	pub fn create_buffer_view_partial(&self, range: &BufferViewRange) -> Result<VulkanBufferView, VulkanError> {
		VulkanBufferView::new_partial(self.buffer.clone(), range)
	}
}

impl Clone for Buffer {
	fn clone(&self) -> Self {
		Self::new(self.device.clone(), self.get_size(), self.staging_buffer.as_ref().map(|b|b.get_address() as *const _), self.usage).unwrap()
	}
}

impl Debug for Buffer {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("Buffer")
		.field("memory", &self.memory)
		.field("buffer", &self.buffer)
		.field("usage", &self.usage)
		.field("staging_buffer", &self.staging_buffer)
		.finish()
	}
}

/// The trait that the struct of uniform must implement
pub trait UniformStructType: Copy + Clone + Sized + Default + Debug + Iterable {}
impl<T> UniformStructType for T where T: Copy + Clone + Sized + Default + Debug + Iterable {}

#[macro_export]
macro_rules! derive_uniform_buffer_type {
	($item: item) => {
		#[repr(C)]
		#[derive(Iterable, Default, Debug, Clone, Copy)]
		$item
	};
}

/// The uniform buffer
#[derive(Debug, Clone)]
pub struct UniformBuffer<U>
where
	U: UniformStructType {
	/// The buffer
	pub buffer: Buffer,

	/// The phantom data that holds the uniform struct type
	_phantom: PhantomData<U>,
}

impl<U> UniformBuffer<U>
where
	U: UniformStructType {
	/// Create the `UniformBuffer`
	pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VulkanError> {
		let def = U::default();
		let buffer = Buffer::new(device.clone(), size_of::<U>() as VkDeviceSize, Some(&def as *const U as *const c_void), VkBufferUsageFlagBits::VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT as VkBufferUsageFlags)?;
		Ok(Self {
			buffer,
			_phantom: PhantomData,
		})
	}

	/// Flush to GPU
	pub fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.buffer.upload_staging_buffer(cmdbuf, 0, self.buffer.get_size())
	}
}

impl<U> AsRef<U> for UniformBuffer<U>
where
	U: UniformStructType {
	fn as_ref(&self) -> &U {
		unsafe{&*(self.buffer.staging_buffer.as_ref().unwrap().get_address() as *const U)}
	}
}

impl<U> AsMut<U> for UniformBuffer<U>
where
	U: UniformStructType {
	fn as_mut(&mut self) -> &mut U {
		unsafe{&mut *(self.buffer.staging_buffer.as_ref().unwrap().get_address() as *mut U)}
	}
}

unsafe impl<U> Send for UniformBuffer<U> where U: UniformStructType {}
unsafe impl<U> Sync for UniformBuffer<U> where U: UniformStructType {}

/// The trait for the `UniformBuffer` to be able to wrap into an object
pub trait GenericUniformBuffer: Debug {
	/// Get the `VkBuffer`
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Get the size of the buffer
	fn get_size(&self) -> VkDeviceSize;

	/// Get the address of the staging buffer
	fn get_staging_buffer_address(&self) -> *mut c_void;

	/// Upload to GPU
	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError>;
}

impl<U> GenericUniformBuffer for UniformBuffer<U>
where
	U: UniformStructType {
	fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.get_vk_buffer()
	}

	fn get_size(&self) -> VkDeviceSize {
		self.buffer.get_size()
	}

	fn get_staging_buffer_address(&self) -> *mut c_void {
		self.buffer.staging_buffer.as_ref().unwrap().get_address()
	}

	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.buffer.upload_staging_buffer(cmdbuf, 0, self.get_size() as VkDeviceSize)
	}
}

/// The trait for the `StorageBuffer` to be able to wrap into an object
pub trait GenericStorageBuffer: Debug {
	/// Get the `VkBuffer`
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Get the size of the buffer
	fn get_size(&self) -> VkDeviceSize;

	/// Get the address of the staging buffer
	fn get_staging_buffer_address(&self) -> *mut c_void;

	/// Upload to GPU
	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError>;
}

/// The trait that the struct of uniform must implement
pub trait StorageBufferStructType: Copy + Clone + Sized + Default + Debug + Iterable {}
impl<T> StorageBufferStructType for T where T: Copy + Clone + Sized + Default + Debug + Iterable {}

#[macro_export]
macro_rules! derive_storage_buffer_type {
	($item: item) => {
		#[repr(C)]
		#[derive(Iterable, Default, Debug, Clone, Copy)]
		$item
	};
}

/// The storage buffer
#[derive(Debug, Clone)]
pub struct StorageBuffer<S>
where
	S: StorageBufferStructType {
	/// The buffer
	pub buffer: Buffer,

	/// The phantom data that holds the uniform struct type
	_phantom: PhantomData<S>,
}

impl<S> StorageBuffer<S>
where
	S: StorageBufferStructType {
	/// Create the `StorageBuffer`
	pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VulkanError> {
		let def = S::default();
		let buffer = Buffer::new(device.clone(), size_of::<S>() as VkDeviceSize, Some(&def as *const S as *const c_void), VkBufferUsageFlagBits::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT as VkBufferUsageFlags)?;
		Ok(Self {
			buffer,
			_phantom: PhantomData,
		})
	}
}

impl<S> AsRef<S> for StorageBuffer<S>
where
	S: StorageBufferStructType {
	fn as_ref(&self) -> &S {
		unsafe{&*(self.buffer.staging_buffer.as_ref().unwrap().get_address() as *const S)}
	}
}

impl<S> AsMut<S> for StorageBuffer<S>
where
	S: StorageBufferStructType {
	fn as_mut(&mut self) -> &mut S {
		unsafe{&mut *(self.buffer.staging_buffer.as_ref().unwrap().get_address() as *mut S)}
	}
}

unsafe impl<S> Send for StorageBuffer<S> where S: StorageBufferStructType {}
unsafe impl<S> Sync for StorageBuffer<S> where S: StorageBufferStructType {}

impl<S> GenericStorageBuffer for StorageBuffer<S>
where
	S: StorageBufferStructType {
	fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.get_vk_buffer()
	}

	fn get_size(&self) -> VkDeviceSize {
		self.buffer.get_size()
	}

	fn get_staging_buffer_address(&self) -> *mut c_void {
		self.buffer.staging_buffer.as_ref().unwrap().get_address()
	}

	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.buffer.upload_staging_buffer(cmdbuf, 0, self.get_size() as VkDeviceSize)
	}
}
