
use crate::prelude::*;
use std::{
	any::Any,
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::size_of,
	sync::{Arc, RwLock},
	vec::IntoIter,
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
	pub staging_buffer: RwLock<Option<StagingBuffer>>,
}

impl Buffer {
	/// Create a new buffer
	/// * If `data` is `None`, `cmdbuf` could be `null()` because no `vkCmdCopyBuffer()` will be issued.
	pub fn new(device: Arc<VulkanDevice>, size: VkDeviceSize, data: Option<*const c_void>, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let buffer = Arc::new(VulkanBuffer::new(device.clone(), size, usage | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT as VkBufferUsageFlags)?);
		let memory = Arc::new(VulkanMemory::new(device.clone(), &buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as VkMemoryPropertyFlags)?);
		memory.bind_vk_buffer(buffer.get_vk_buffer())?;
		let ret = Self {
			device,
			memory,
			buffer,
			usage,
			staging_buffer: RwLock::new(None),
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
	pub fn ensure_staging_buffer(&self) -> Result<(), VulkanError> {
		let mut lock = self.staging_buffer.write().unwrap();
		if lock.is_none() {
			*lock = Some(StagingBuffer::new(self.device.clone(), self.buffer.get_size())?);
		}
		Ok(())
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&self) {
		let mut lock = self.staging_buffer.write().unwrap();
		*lock = None;
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
	pub fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError> {
		self.ensure_staging_buffer()?;
		Ok(self.staging_buffer.read().unwrap().as_ref().unwrap().get_address())
	}

	/// Update new data to the buffer
	///
	/// # Safety
	///
	/// You must provide a valid pointer `data`, otherwise the behavior of this function is undefined.
	pub unsafe fn set_staging_data(&self, data: *const c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		self.ensure_staging_buffer()?;
		self.staging_buffer.write().unwrap().as_mut().unwrap().set_data(data, offset, size)?;
		Ok(())
	}

	/// Retrieve the data from the staging buffer
	///
	/// # Safety
	///
	/// You must provide a valid pointer `data`, otherwise the behavior of this function is undefined.
	pub unsafe fn get_staging_data(&self, data: *mut c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		let lock = self.staging_buffer.read().unwrap();
		if let Some(ref staging_buffer) = *lock {
			staging_buffer.get_data(data, offset, size)
		} else {
			Err(VulkanError::NoStagingBuffer)
		}
	}

	/// Upload the data from the staging buffer
	pub fn upload_staging_buffer(&self, cmdbuf: VkCommandBuffer, offset: VkDeviceSize, size: VkDeviceSize) -> Result<(), VulkanError> {
		let lock = self.staging_buffer.read().unwrap();
		if let Some(ref staging_buffer) = *lock {
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
		let lock = self.staging_buffer.read().unwrap();
		if let Some(ref staging_buffer) = *lock {
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
	pub fn download_staging_buffer(&self, cmdbuf: VkCommandBuffer, offset: VkDeviceSize, size: VkDeviceSize) -> Result<(), VulkanError> {
		self.ensure_staging_buffer()?;
		let copy_region = VkBufferCopy {
			srcOffset: offset,
			dstOffset: offset,
			size: size as VkDeviceSize,
		};
		self.device.vkcore.vkCmdCopyBuffer(cmdbuf, self.buffer.get_vk_buffer(), self.staging_buffer.read().unwrap().as_ref().unwrap().get_vk_buffer(), 1, &copy_region)?;
		Ok(())
	}

	/// Download the data to the staging buffer
	pub fn download_staging_buffer_multi(&self, cmdbuf: VkCommandBuffer, regions: &[BufferRegion]) -> Result<(), VulkanError> {
		self.ensure_staging_buffer()?;
		let copy_regions: Vec<VkBufferCopy> = regions.iter().map(|r|VkBufferCopy {
			srcOffset: r.offset,
			dstOffset: r.offset,
			size: r.size as VkDeviceSize,
		}).collect();
		self.device.vkcore.vkCmdCopyBuffer(cmdbuf, self.buffer.get_vk_buffer(), self.staging_buffer.read().unwrap().as_ref().unwrap().get_vk_buffer(), copy_regions.len() as u32, copy_regions.as_ptr())?;
		Ok(())
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
		Self::new(self.device.clone(), self.get_size(), self.staging_buffer.read().unwrap().as_ref().map(|b|b.get_address() as *const _), self.usage).unwrap()
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
pub trait UniformStructType: Copy + Clone + Sized + Default + Send + Sync + Debug + Iterable + Any {}
impl<T> UniformStructType for T where T: Copy + Clone + Sized + Default + Send + Sync + Debug + Iterable + Any {}

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

	/// The iterable struct that holds the uniform struct type
	iterable: U,
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
			iterable: def,
		})
	}

	/// Create the staging buffer if not exist
	pub fn ensure_staging_buffer(&self) -> Result<(), VulkanError> {
		self.buffer.ensure_staging_buffer()
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&self) {
		self.buffer.discard_staging_buffer()
	}

	/// Get the address of the staging buffer memory data
	pub fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError> {
		self.buffer.get_staging_buffer_address()
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
		unsafe{&*(self.get_staging_buffer_address().unwrap() as *const U)}
	}
}

impl<U> AsMut<U> for UniformBuffer<U>
where
	U: UniformStructType {
	fn as_mut(&mut self) -> &mut U {
		unsafe{&mut *(self.get_staging_buffer_address().unwrap() as *mut U)}
	}
}

unsafe impl<U> Send for UniformBuffer<U> where U: UniformStructType {}
unsafe impl<U> Sync for UniformBuffer<U> where U: UniformStructType {}

/// The trait for the `UniformBuffer` to be able to wrap into an object
pub trait GenericUniformBuffer: IterableDataAttrib + Debug + Any + Send + Sync {
	/// Get the `VkBuffer`
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Get the size of the buffer
	fn get_size(&self) -> VkDeviceSize;

	/// Get the address of the staging buffer
	fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError>;

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

	fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError> {
		self.get_staging_buffer_address()
	}

	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.buffer.upload_staging_buffer(cmdbuf, 0, self.get_size() as VkDeviceSize)
	}
}

impl<U> IterableDataAttrib for UniformBuffer<U>
where
	U: UniformStructType {
	fn iter_members(&self) -> IntoIter<(&'static str, &dyn Any)> {
		self.iterable.iter()
	}
}

/// The trait for the `StorageBuffer` to be able to wrap into an object
pub trait GenericStorageBuffer: IterableDataAttrib + Debug + Any + Send + Sync {
	/// Get the `VkBuffer`
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Get the size of the buffer
	fn get_size(&self) -> VkDeviceSize;

	/// Get the address of the staging buffer
	fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError>;

	/// Upload to GPU
	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError>;
}

/// The trait that the struct of uniform must implement
pub trait StorageBufferStructType: Copy + Clone + Sized + Default + Send + Sync + Debug + Iterable + Any {}
impl<T> StorageBufferStructType for T where T: Copy + Clone + Sized + Default + Send + Sync + Debug + Iterable + Any {}

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

	/// The iterable struct that holds the storage buffer struct type
	iterable: S,
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
			iterable: def,
		})
	}

	/// Create the staging buffer if not exist
	pub fn ensure_staging_buffer(&self) -> Result<(), VulkanError> {
		self.buffer.ensure_staging_buffer()
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&self) {
		self.buffer.discard_staging_buffer()
	}

	/// Get the address of the staging buffer memory data
	pub fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError> {
		self.buffer.get_staging_buffer_address()
	}

	/// Flush to GPU
	pub fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.buffer.upload_staging_buffer(cmdbuf, 0, self.buffer.get_size())
	}
}

impl<S> AsRef<S> for StorageBuffer<S>
where
	S: StorageBufferStructType {
	fn as_ref(&self) -> &S {
		unsafe{&*(self.get_staging_buffer_address().unwrap() as *const S)}
	}
}

impl<S> AsMut<S> for StorageBuffer<S>
where
	S: StorageBufferStructType {
	fn as_mut(&mut self) -> &mut S {
		unsafe{&mut *(self.get_staging_buffer_address().unwrap() as *mut S)}
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

	fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError> {
		self.get_staging_buffer_address()
	}

	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.buffer.upload_staging_buffer(cmdbuf, 0, self.get_size() as VkDeviceSize)
	}
}

impl<S> IterableDataAttrib for StorageBuffer<S>
where
	S: StorageBufferStructType {
	fn iter_members(&self) -> IntoIter<(&'static str, &dyn Any)> {
		self.iterable.iter()
	}
}
