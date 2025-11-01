
use crate::prelude::*;
use std::{
	any::Any,
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::size_of,
	ops::{Index, IndexMut, Range, RangeFrom, RangeTo, RangeFull, RangeInclusive, RangeToInclusive},
	slice::from_raw_parts_mut,
	sync::{
		Arc,
		RwLock,
		RwLockWriteGuard,
	},
	vec::IntoIter,
};

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
	pub fn ensure_staging_buffer<'a>(&'a self) -> Result<RwLockWriteGuard<'a, Option<StagingBuffer>>, VulkanError> {
		let mut lock = self.staging_buffer.write().unwrap();
		if lock.is_none() {
			*lock = Some(StagingBuffer::new(self.device.clone(), self.buffer.get_size())?);
		}
		Ok(lock)
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

	/// Map staging buffer as slice
	pub fn map_staging_buffer_as_slice_locked<'a, T>(&'a self) -> Result<BufferMapGuard<'a, T>, VulkanError>
	where
		T: Sized + Clone + Copy {
		BufferMapGuard::new(self.ensure_staging_buffer()?, self.get_size() as usize)
	}

	/// Get the address of the staging buffer memory data
	pub fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError> {
		let lock = self.ensure_staging_buffer()?;
		Ok(lock.as_ref().unwrap().get_address())
	}

	/// Update new data to the buffer
	///
	/// # Safety
	///
	/// You must provide a valid pointer `data`, otherwise the behavior of this function is undefined.
	pub unsafe fn set_staging_data(&self, data: *const c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		let lock = self.ensure_staging_buffer()?;
		lock.as_ref().unwrap().set_data(data, offset, size)?;
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
		let lock = self.ensure_staging_buffer()?;
		let copy_region = VkBufferCopy {
			srcOffset: offset,
			dstOffset: offset,
			size: size as VkDeviceSize,
		};
		self.device.vkcore.vkCmdCopyBuffer(cmdbuf, self.buffer.get_vk_buffer(), lock.as_ref().unwrap().get_vk_buffer(), 1, &copy_region)?;
		Ok(())
	}

	/// Download the data to the staging buffer
	pub fn download_staging_buffer_multi(&self, cmdbuf: VkCommandBuffer, regions: &[BufferRegion]) -> Result<(), VulkanError> {
		let lock = self.ensure_staging_buffer()?;
		let copy_regions: Vec<VkBufferCopy> = regions.iter().map(|r|VkBufferCopy {
			srcOffset: r.offset,
			dstOffset: r.offset,
			size: r.size as VkDeviceSize,
		}).collect();
		self.device.vkcore.vkCmdCopyBuffer(cmdbuf, self.buffer.get_vk_buffer(), lock.as_ref().unwrap().get_vk_buffer(), copy_regions.len() as u32, copy_regions.as_ptr())?;
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


/// The typed map
#[derive(Debug)]
pub struct BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	/// The lock guard
	lock_guard: RwLockWriteGuard<'a, Option<StagingBuffer>>,

	/// The slice of items
	slice: &'a mut [T],
}

impl<'a, T> BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	pub fn new(lock_guard: RwLockWriteGuard<'a, Option<StagingBuffer>>, size: usize) -> Result<Self, VulkanError> {
		let address = lock_guard.as_ref().unwrap().get_address();
		let len = size / size_of::<T>();
		let slice = unsafe {from_raw_parts_mut(address as *mut T, len)};
		Ok(Self {
			lock_guard,
			slice,
		})
	}

	/// Operate the mapped memory as a slice
	pub fn as_slice(&self) -> &[T] {
		self.slice
	}

	/// Operate the mapped memory as a mutable slice
	pub fn as_slice_mut(&mut self) -> &mut [T] {
		self.slice
	}
}

impl<'a, T> Index<usize> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = T;
	fn index(&self, index: usize) -> &T {
		&self.slice[index]
	}
}

impl<'a, T> IndexMut<usize> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, index: usize) -> &mut T {
		&mut self.slice[index]
	}
}

impl<'a, T> Index<Range<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: Range<usize>) -> &[T] {
		&self.slice[range.start..range.end]
	}
}

impl<'a, T> IndexMut<Range<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: Range<usize>) -> &mut [T] {
		&mut self.slice[range.start..range.end]
	}
}

impl<'a, T> Index<RangeFrom<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: RangeFrom<usize>) -> &[T] {
		&self.slice[range.start..]
	}
}

impl<'a, T> IndexMut<RangeFrom<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut [T] {
		&mut self.slice[range.start..]
	}
}

impl<'a, T> Index<RangeTo<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: RangeTo<usize>) -> &[T] {
		&self.slice[..range.end]
	}
}

impl<'a, T> IndexMut<RangeTo<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: RangeTo<usize>) -> &mut [T] {
		&mut self.slice[..range.end]
	}
}

impl<'a, T> Index<RangeFull> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, _: RangeFull) -> &[T] {
		&self.slice[..]
	}
}

impl<'a, T> IndexMut<RangeFull> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, _: RangeFull) -> &mut [T] {
		&mut self.slice[..]
	}
}

impl<'a, T> Index<RangeInclusive<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: RangeInclusive<usize>) -> &[T] {
		&self.slice[*range.start()..=*range.end()]
	}
}

impl<'a, T> IndexMut<RangeInclusive<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: RangeInclusive<usize>) -> &mut [T] {
		&mut self.slice[*range.start()..=*range.end()]
	}
}

impl<'a, T> Index<RangeToInclusive<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: RangeToInclusive<usize>) -> &[T] {
		&self.slice[..=range.end]
	}
}

impl<'a, T> IndexMut<RangeToInclusive<usize>> for BufferMapGuard<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: RangeToInclusive<usize>) -> &mut [T] {
		&mut self.slice[..=range.end]
	}
}

/// The trait that the struct of uniform must implement
pub trait UniformStructType: Copy + Clone + Sized + Default + Send + Sync + Debug + FFIStruct + Any {}
impl<T> UniformStructType for T where T: Copy + Clone + Sized + Default + Send + Sync + Debug + FFIStruct + Any {}

#[macro_export]
macro_rules! derive_uniform_buffer_type {
	($item: item) => {
		#[ffi_struct]
		#[derive(Default, Debug, Clone, Copy)]
		#[size_of_type (Vec1 = 4, Vec2 = 8, Vec3 = 12, Vec4 = 16)]
		#[align_of_type(Vec1 = 4, Vec2 = 8, Vec3 = 16, Vec4 = 16)]
		#[size_of_type (Mat1 = 4, Mat2 = 16, Mat3 = 48, Mat4 = 64)]
		#[align_of_type(Mat1 = 4, Mat2 = 8, Mat3 = 16, Mat4 = 16)]
		#[size_of_type (Mat1x1 = 4, Mat2x2 = 16, Mat3x3 = 48, Mat4x4 = 64)]
		#[align_of_type(Mat1x1 = 4, Mat2x2 = 8, Mat3x3 = 16, Mat4x4 = 16)]
		#[size_of_type (Mat1x2 = 8, Mat1x3 = 12, Mat1x4 = 16)]
		#[align_of_type(Mat1x2 = 4, Mat1x3 = 4, Mat1x4 = 4)]
		#[size_of_type (Mat2x1 = 8, Mat2x3 = 24, Mat2x4 = 32)]
		#[align_of_type(Mat2x1 = 8, Mat2x3 = 8, Mat2x4 = 8)]
		#[size_of_type (Mat2x3 = 24, Mat2x4 = 32)]
		#[align_of_type(Mat2x3 = 8, Mat2x4 = 8)]
		#[size_of_type (Mat3x2 = 32, Mat3x4 = 64)]
		#[align_of_type(Mat3x2 = 16, Mat3x4 = 16)]
		#[size_of_type (Mat4x2 = 32, Mat4x3 = 48)]
		#[align_of_type(Mat4x2 = 16, Mat4x3 = 16)]
		#[size_of_type (DVec1 = 8, DVec2 = 16, DVec3 = 24, DVec4 = 32)]
		#[align_of_type(DVec1 = 8, DVec2 = 16, DVec3 = 32, DVec4 = 32)]
		#[size_of_type (DMat1 = 8, DMat2 = 32, DMat3 = 96, DMat4 = 128)]
		#[align_of_type(DMat1 = 8, DMat2 = 16, DMat3 = 32, DMat4 = 32)]
		#[size_of_type (DMat1x1 = 8, DMat2x2 = 32, DMat3x3 = 96, DMat4x4 = 128)]
		#[align_of_type(DMat1x1 = 8, DMat2x2 = 16, DMat3x3 = 32, DMat4x4 = 32)]
		#[size_of_type (DMat1x2 = 16, DMat1x3 = 24, DMat1x4 = 32)]
		#[align_of_type(DMat1x2 = 8, DMat1x3 = 8, DMat1x4 = 8)]
		#[size_of_type (DMat2x1 = 16, DMat2x3 = 48, DMat2x4 = 64)]
		#[align_of_type(DMat2x1 = 16, DMat2x3 = 16, DMat2x4 = 16)]
		#[size_of_type (DMat2x3 = 48, DMat2x4 = 64)]
		#[align_of_type(DMat2x3 = 16, DMat2x4 = 32)]
		#[size_of_type (DMat3x2 = 64, DMat3x4 = 128)]
		#[align_of_type(DMat3x2 = 32, DMat3x4 = 32)]
		#[size_of_type (DMat4x2 = 64, DMat4x3 = 96)]
		#[align_of_type(DMat4x2 = 32, DMat4x3 = 32)]
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
	pub fn new(device: Arc<VulkanDevice>, initial_value: Option<U>) -> Result<Self, VulkanError> {
		let def = initial_value.unwrap_or_default();
		let buffer = Buffer::new(device.clone(), size_of::<U>() as VkDeviceSize, Some(&def as *const U as *const c_void), VkBufferUsageFlagBits::VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT as VkBufferUsageFlags)?;
		Ok(Self {
			buffer,
			iterable: U::default(),
		})
	}

	/// Create the staging buffer if not exist
	pub fn ensure_staging_buffer<'a>(&'a self) -> Result<RwLockWriteGuard<'a, Option<StagingBuffer>>, VulkanError> {
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
pub trait GenericUniformBuffer: Debug + Any + Send + Sync {
	/// Get the `VkBuffer`
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Iterate through the members of the generic type `U`
	fn iter_members(&self) -> IntoIter<(&'static str, MemberInfo)>;

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

	fn iter_members(&self) -> IntoIter<(&'static str, MemberInfo)> {
		self.iterable.iter_members()
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

/// The trait for the `StorageBuffer` to be able to wrap into an object
pub trait GenericStorageBuffer: Debug + Any + Send + Sync {
	/// Get the `VkBuffer`
	fn get_vk_buffer(&self) -> VkBuffer;

	/// Iterate through the members of the generic type `S`
	fn iter_members(&self) -> IntoIter<(&'static str, MemberInfo)>;

	/// Get the size of the buffer
	fn get_size(&self) -> VkDeviceSize;

	/// Get the address of the staging buffer
	fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError>;

	/// Upload to GPU
	fn flush(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError>;
}

/// The trait that the struct of uniform must implement
pub trait StorageBufferStructType: Copy + Clone + Sized + Default + Send + Sync + Debug + FFIStruct + Any {}
impl<T> StorageBufferStructType for T where T: Copy + Clone + Sized + Default + Send + Sync + Debug + FFIStruct + Any {}

#[macro_export]
macro_rules! derive_storage_buffer_type {
	($item: item) => {
		#[repr(C)]
		#[derive(Default, Debug, Clone, Copy)]
		#[size_of_type (Vec1 = 4, Vec2 = 8, Vec3 = 12, Vec4 = 16)]
		#[align_of_type(Vec1 = 4, Vec2 = 4, Vec3 = 4, Vec4 = 4)]
		#[size_of_type (Mat1 = 4, Mat2 = 16, Mat3 = 36, Mat4 = 64)]
		#[align_of_type(Mat1 = 4, Mat2 = 4, Mat3 = 4, Mat4 = 4)]
		#[size_of_type (Mat1x1 = 4, Mat2x2 = 16, Mat3x3 = 36, Mat4x4 = 64)]
		#[align_of_type(Mat1x1 = 4, Mat2x2 = 4, Mat3x3 = 4, Mat4x4 = 4)]
		#[size_of_type (Mat1x2 = 8, Mat1x3 = 12, Mat1x4 = 16)]
		#[align_of_type(Mat1x2 = 4, Mat1x3 = 4, Mat1x4 = 4)]
		#[size_of_type (Mat2x1 = 8, Mat2x3 = 24, Mat2x4 = 32)]
		#[align_of_type(Mat2x1 = 4, Mat2x3 = 4, Mat2x4 = 4)]
		#[size_of_type (Mat2x3 = 24, Mat2x4 = 32)]
		#[align_of_type(Mat2x3 = 4, Mat2x4 = 4)]
		#[size_of_type (Mat3x2 = 24, Mat3x4 = 48)]
		#[align_of_type(Mat3x2 = 4, Mat3x4 = 4)]
		#[size_of_type (Mat4x2 = 32, Mat4x3 = 48)]
		#[align_of_type(Mat4x2 = 4, Mat4x3 = 4)]
		#[size_of_type (DVec1 = 8, DVec2 = 16, DVec3 = 24, DVec4 = 32)]
		#[align_of_type(DVec1 = 4, DVec2 = 4, DVec3 = 4, DVec4 = 4)]
		#[size_of_type (DMat1 = 8, DMat2 = 32, DMat3 = 72, DMat4 = 128)]
		#[align_of_type(DMat1 = 4, DMat2 = 4, DMat3 = 4, DMat4 = 4)]
		#[size_of_type (DMat1x1 = 8, DMat2x2 = 32, DMat3x3 = 72, DMat4x4 = 128)]
		#[align_of_type(DMat1x1 = 4, DMat2x2 = 4, DMat3x3 = 4, DMat4x4 = 4)]
		#[size_of_type (DMat1x2 = 16, DMat1x3 = 24, DMat1x4 = 32)]
		#[align_of_type(DMat1x2 = 4, DMat1x3 = 4, DMat1x4 = 4)]
		#[size_of_type (DMat2x1 = 16, DMat2x3 = 48, DMat2x4 = 64)]
		#[align_of_type(DMat2x1 = 4, DMat2x3 = 4, DMat2x4 = 4)]
		#[size_of_type (DMat2x3 = 48, DMat2x4 = 64)]
		#[align_of_type(DMat2x3 = 4, DMat2x4 = 4)]
		#[size_of_type (DMat3x2 = 48, DMat3x4 = 96)]
		#[align_of_type(DMat3x2 = 4, DMat3x4 = 4)]
		#[size_of_type (DMat4x2 = 64, DMat4x3 = 96)]
		#[align_of_type(DMat4x2 = 4, DMat4x3 = 4)]
		#[size_of_type (bool = 4)]
		#[align_of_type(bool = 4)]
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
	pub fn new(device: Arc<VulkanDevice>, initial_value: Option<S>) -> Result<Self, VulkanError> {
		let def = initial_value.unwrap_or_default();
		let buffer = Buffer::new(device.clone(), size_of::<S>() as VkDeviceSize, Some(&def as *const S as *const c_void), VkBufferUsageFlagBits::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT as VkBufferUsageFlags)?;
		Ok(Self {
			buffer,
			iterable: S::default(),
		})
	}

	/// Create the staging buffer if not exist
	pub fn ensure_staging_buffer<'a>(&'a self) -> Result<RwLockWriteGuard<'a, Option<StagingBuffer>>, VulkanError> {
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

	fn iter_members(&self) -> IntoIter<(&'static str, MemberInfo)> {
		self.iterable.iter_members()
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

#[macro_export]
macro_rules! get_generic_uniform_buffer_cache {
	($gub:expr,$t:ty) => (&mut *($gub.get_staging_buffer_address()? as *mut $t))
}

#[macro_export]
macro_rules! get_generic_storage_buffer_cache {
	($gsb:expr,$t:ty) => (&mut *($gsb.get_staging_buffer_address()? as *mut $t))
}
