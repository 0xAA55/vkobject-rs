
#![allow(clippy::uninit_vec)]
#![allow(clippy::too_many_arguments)]
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	io::{self, ErrorKind},
	marker::PhantomData,
	mem::{MaybeUninit, size_of},
	ops::{Index, IndexMut, Range, RangeFrom, RangeTo, RangeFull, RangeInclusive, RangeToInclusive},
	ptr::{copy, null, null_mut},
	slice::from_raw_parts_mut,
	sync::{
		Arc,
		Mutex,
	},
};

/// The error for almost all of the crate's `Result<>`
#[derive(Debug, Clone)]
pub enum VulkanError {
	VkError(VkError),
	IOError((String, ErrorKind)),
	ChooseGpuFailed,
	NoGoodQueueForSurface(&'static str),
	NoGoodDepthStencilFormat,
	CommandPoolIsInUse,
	NoIdleCommandPools,
	NoIdleDeviceQueues,
	NoSuitableMemoryType,
	ImageTypeSizeNotMatch(String),
	ImagePixelFormatNotSupported,
	ShaderCompilationError(String),
	ShaderParseIdUnknown(String),
	ShaderParseTypeUnknown(String),
	ShaderParseError(Arc<rspirv::binary::ParseState>),
	MissingShaderInputs(String),
	ShaderInputTypeMismatch(String),
	ShaderInputLengthMismatch(String),
	ShaderInputTypeUnsupported(String),
}

impl From<VkError> for VulkanError {
	fn from(e: VkError) -> Self {
		Self::VkError(e)
	}
}

impl From<io::Error> for VulkanError {
	fn from(e: io::Error) -> Self {
		Self::IOError((format!("{e:?}"), e.kind()))
	}
}

impl From<rspirv::binary::ParseState> for VulkanError {
	fn from(s: rspirv::binary::ParseState) -> Self {
		Self::ShaderParseError(Arc::new(s))
	}
}

#[cfg(feature = "shaderc")]
impl From<shaderc::Error> for VulkanError {
	fn from(e: shaderc::Error) -> Self {
		match e {
			shaderc::Error::CompilationError(_, desc) => Self::ShaderCompilationError(desc),
			_ => Self::ShaderCompilationError(format!("{e:?}")),
		}
	}
}

impl VulkanError {
	pub fn is_vkerror(&self) -> Option<&VkError> {
		if let Self::VkError(ve) = self {
			Some(ve)
		} else {
			None
		}
	}

	pub fn is_shader_error(&self) -> Option<&String> {
		if let Self::ShaderCompilationError(se) = self {
			Some(se)
		} else {
			None
		}
	}
}

/// The wrapper for the `VkPipelineCache`
pub struct VulkanPipelineCache {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The `VkPipelineCache`
	pipeline_cache: VkPipelineCache,
}

impl VulkanPipelineCache {
	/// Create the `VulkanPipelineCache`
	pub fn new(device: Arc<VulkanDevice>, initial_data: Option<&[u8]>) -> Result<Self, VulkanError> {
		let (data_size, data_ptr) = if let Some(data) = initial_data {
			(data.len(), data.as_ptr())
		} else {
			(0, null())
		};
		let pipeline_cache_ci = VkPipelineCacheCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			initialDataSize: data_size,
			pInitialData: data_ptr as *const c_void,
		};
		let mut pipeline_cache = null();
		device.vkcore.vkCreatePipelineCache(device.get_vk_device(), &pipeline_cache_ci, null(), &mut pipeline_cache)?;
		Ok(Self {
			device,
			pipeline_cache,
		})
	}

	/// Dump the pipeline cache to binary. The binary could be used for the next creation of the `VulkanPipelineCache` for faster pipeline creation
	pub fn dump_cache(&self) -> Result<Vec<u8>, VulkanError> {
		let vkdevice = self.device.get_vk_device();
		let mut size = 0;
		self.device.vkcore.vkGetPipelineCacheData(vkdevice, self.pipeline_cache, &mut size, null_mut())?;
		if size == 0 {
			Ok(Vec::new())
		} else {
			let mut data: Vec<u8> = Vec::with_capacity(size);
			self.device.vkcore.vkGetPipelineCacheData(vkdevice, self.pipeline_cache, &mut size, data.as_mut_ptr() as *mut c_void)?;
			unsafe {data.set_len(size)};
			Ok(data)
		}
	}

	/// Get the `VkPipelineCache`
	pub(crate) fn get_vk_pipeline_cache(&self) -> VkPipelineCache {
		self.pipeline_cache
	}
}

impl Debug for VulkanPipelineCache {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanPipelineCache")
		.field("pipeline_cache", &self.pipeline_cache)
		.finish()
	}
}

impl Drop for VulkanPipelineCache {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyPipelineCache(self.device.get_vk_device(), self.pipeline_cache, null()).unwrap()
	}
}

unsafe impl Send for VulkanPipelineCache {}
unsafe impl Sync for VulkanPipelineCache {}

/// The wrapper for the `VkSemaphore`
pub struct VulkanSemaphore {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The semaphore handle
	semaphore: VkSemaphore,

	/// For the timeline semaphore, this is the timeline value
	pub(crate) timeline: u64,
}

unsafe impl Send for VulkanSemaphore {}
unsafe impl Sync for VulkanSemaphore {}

impl VulkanSemaphore {
	/// Create a new binary semaphore
	pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VulkanError> {
		let ci = VkSemaphoreCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			pNext: null(),
			flags: 0,
		};
		let mut semaphore: VkSemaphore = null();
		device.vkcore.vkCreateSemaphore(device.get_vk_device(), &ci, null(), &mut semaphore)?;
		Ok(Self{
			device,
			semaphore,
			timeline: 0,
		})
	}

	/// Create a new timeline semaphore
	pub fn new_timeline(device: Arc<VulkanDevice>, initial_value: u64) -> Result<Self, VulkanError> {
		let ci_next = VkSemaphoreTypeCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
			pNext: null(),
			semaphoreType: VkSemaphoreType::VK_SEMAPHORE_TYPE_TIMELINE,
			initialValue: initial_value,
		};
		let ci = VkSemaphoreCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			pNext: &ci_next as *const VkSemaphoreTypeCreateInfo as *const c_void,
			flags: 0,
		};
		let mut semaphore: VkSemaphore = null();
		device.vkcore.vkCreateSemaphore(device.get_vk_device(), &ci, null(), &mut semaphore)?;
		Ok(Self{
			device,
			semaphore,
			timeline: initial_value,
		})
	}

	/// Signal the semaphore
	pub fn signal(&self, value: u64) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let signal_i = VkSemaphoreSignalInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
			pNext: null(),
			semaphore: self.semaphore,
			value,
		};
		vkcore.vkSignalSemaphore(self.device.get_vk_device(), &signal_i)?;
		Ok(())
	}

	/// Get the `VkSemaphore`
	pub(crate) fn get_vk_semaphore(&self) -> VkSemaphore {
		self.semaphore
	}

	/// Wait for the semaphore
	pub fn wait(&self, timeout: u64) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let vk_device = self.device.get_vk_device();
		let semaphores = [self.semaphore];
		let timelines = [self.timeline];
		let wait_i = VkSemaphoreWaitInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
			pNext: null(),
			flags: 0,
			semaphoreCount: 1,
			pSemaphores: semaphores.as_ptr(),
			pValues: timelines.as_ptr(),
		};
		vkcore.vkWaitSemaphores(vk_device, &wait_i, timeout)?;
		Ok(())
	}

	/// Wait for the semaphore
	pub fn wait_vk(device: &VulkanDevice, semaphore: VkSemaphore, timeline: u64, timeout: u64) -> Result<(), VulkanError> {
		let vkcore = device.vkcore.clone();
		let vk_device = device.get_vk_device();
		let semaphores = [semaphore];
		let timelines = [timeline];
		let wait_i = VkSemaphoreWaitInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
			pNext: null(),
			flags: 0,
			semaphoreCount: 1,
			pSemaphores: semaphores.as_ptr(),
			pValues: timelines.as_ptr(),
		};
		vkcore.vkWaitSemaphores(vk_device, &wait_i, timeout)?;
		Ok(())
	}

	/// Wait for multiple semaphores
	pub fn wait_multi(semaphores: &[Self], timeout: u64, any: bool) -> Result<(), VulkanError> {
		if semaphores.is_empty() {
			Ok(())
		} else {
			let vkcore = semaphores[0].device.vkcore.clone();
			let vk_device = semaphores[0].device.get_vk_device();
			let timelines: Vec<u64> = semaphores.iter().map(|s|s.timeline).collect();
			let semaphores: Vec<VkSemaphore> = semaphores.iter().map(|s|s.get_vk_semaphore()).collect();
			let wait_i = VkSemaphoreWaitInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
				pNext: null(),
				flags: if any {VkSemaphoreWaitFlagBits::VK_SEMAPHORE_WAIT_ANY_BIT as VkSemaphoreWaitFlags} else {0},
				semaphoreCount: semaphores.len() as u32,
				pSemaphores: semaphores.as_ptr(),
				pValues: timelines.as_ptr(),
			};
			vkcore.vkWaitSemaphores(vk_device, &wait_i, timeout)?;
			Ok(())
		}
	}

	/// Wait for multiple semaphores
	pub fn wait_multi_vk(device: &VulkanDevice, semaphores: &[VkSemaphore], timelines: &[u64], timeout: u64, any: bool) -> Result<(), VulkanError> {
		if semaphores.is_empty() {
			Ok(())
		} else {
			let vkcore = device.vkcore.clone();
			let vk_device = device.get_vk_device();
			let wait_i = VkSemaphoreWaitInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
				pNext: null(),
				flags: if any {VkSemaphoreWaitFlagBits::VK_SEMAPHORE_WAIT_ANY_BIT as VkSemaphoreWaitFlags} else {0},
				semaphoreCount: semaphores.len() as u32,
				pSemaphores: semaphores.as_ptr(),
				pValues: timelines.as_ptr(),
			};
			vkcore.vkWaitSemaphores(vk_device, &wait_i, timeout)?;
			Ok(())
		}
	}
}

impl Debug for VulkanSemaphore {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanSemaphore")
		.field("semaphore", &self.semaphore)
		.field("timeline", &self.timeline)
		.finish()
	}
}

impl Drop for VulkanSemaphore {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkDestroySemaphore(self.device.get_vk_device(), self.semaphore, null()).unwrap();
	}
}

/// The wrapper for the `VkFence`
pub struct VulkanFence {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The fence handle
	fence: VkFence,
}

unsafe impl Send for VulkanFence {}
unsafe impl Sync for VulkanFence {}

impl VulkanFence {
	/// Create a new fence
	pub fn new(device: Arc<VulkanDevice>) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let ci = VkFenceCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			pNext: null(),
			flags: 0,
		};
		let mut fence: VkFence = null();
		vkcore.vkCreateFence(device.get_vk_device(), &ci, null(), &mut fence)?;
		Ok(Self{
			device,
			fence,
		})
	}

	/// Get the `VkFence`
	pub(crate) fn get_vk_fence(&self) -> VkFence {
		self.fence
	}

	/// Check if the fence is signaled or not
	pub fn is_signaled(&self) -> Result<bool, VulkanError> {
		let vkcore = self.device.vkcore.clone();
		match vkcore.vkGetFenceStatus(self.device.get_vk_device(), self.fence) {
			Ok(_) => Ok(true),
			Err(e) => match e {
				VkError::VkNotReady(_) => Ok(false),
				others => Err(VulkanError::VkError(others)),
			}
		}
	}

	/// Check if the fence is signaled or not
	pub fn is_signaled_vk(device: &VulkanDevice, fence: VkFence) -> Result<bool, VulkanError> {
		let vkcore = device.vkcore.clone();
		match vkcore.vkGetFenceStatus(device.get_vk_device(), fence) {
			Ok(_) => Ok(true),
			Err(e) => match e {
				VkError::VkNotReady(_) => Ok(false),
				others => Err(VulkanError::VkError(others)),
			}
		}
	}

	/// Unsignal the fence
	pub fn unsignal(&self) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let fences = [self.fence];
		Ok(vkcore.vkResetFences(self.device.get_vk_device(), 1, fences.as_ptr())?)
	}

	/// Unsignal the fence
	pub fn unsignal_multi(fences: &[Self]) -> Result<(), VulkanError> {
		if fences.is_empty() {
			Ok(())
		} else {
			let vkcore = &fences[0].device.vkcore;
			let vkdevice = fences[0].device.get_vk_device();
			let fences: Vec<VkFence> = fences.iter().map(|f|f.get_vk_fence()).collect();
			Ok(vkcore.vkResetFences(vkdevice, fences.len() as u32, fences.as_ptr())?)
		}
	}

	/// Unsignal the fence
	pub fn unsignal_multi_vk(device: &VulkanDevice, fences: &[VkFence]) -> Result<(), VulkanError> {
		if fences.is_empty() {
			Ok(())
		} else {
			let vkcore = device.vkcore.clone();
			Ok(vkcore.vkResetFences(device.get_vk_device(), fences.len() as u32, fences.as_ptr())?)
		}
	}

	/// Wait for the fence to be signaled
	pub fn wait(&self, timeout: u64) -> Result<(), VulkanError> {
		let vk_device = self.device.get_vk_device();
		let fences = [self.fence];
		let vkcore = self.device.vkcore.clone();
		vkcore.vkWaitForFences(vk_device, 1, fences.as_ptr(), 0, timeout)?;
		Ok(())
	}

	/// Wait for the fence to be signaled
	pub fn wait_vk(device: &VulkanDevice, fence: VkFence, timeout: u64) -> Result<(), VulkanError> {
		let vk_device = device.get_vk_device();
		let fences = [fence];
		let vkcore = device.vkcore.clone();
		vkcore.vkWaitForFences(vk_device, 1, fences.as_ptr(), 0, timeout)?;
		Ok(())
	}

	/// Wait for multiple fences to be signaled
	pub fn wait_multi(fences: &[Self], timeout: u64, any: bool) -> Result<(), VulkanError> {
		if fences.is_empty() {
			Ok(())
		} else {
			let vkcore = fences[0].device.vkcore.clone();
			let vk_device = fences[0].device.get_vk_device();
			let fences: Vec<VkFence> = fences.iter().map(|f|f.get_vk_fence()).collect();
			vkcore.vkWaitForFences(vk_device, fences.len() as u32, fences.as_ptr(), if any {0} else {1}, timeout)?;
			Ok(())
		}
	}

	/// Wait for multiple fences to be signaled
	pub fn wait_multi_vk(device: &VulkanDevice, fences: &[VkFence], timeout: u64, any: bool) -> Result<(), VulkanError> {
		if fences.is_empty() {
			Ok(())
		} else {
			let vkcore = device.vkcore.clone();
			let vk_device = device.get_vk_device();
			vkcore.vkWaitForFences(vk_device, fences.len() as u32, fences.as_ptr(), if any {0} else {1}, timeout)?;
			Ok(())
		}
	}
}

impl Debug for VulkanFence {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanFence")
		.field("fence", &self.fence)
		.finish()
	}
}

impl Drop for VulkanFence {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkDestroyFence(self.device.get_vk_device(), self.fence, null()).unwrap();
	}
}

/// The memory object that temporarily stores the `VkDeviceMemory`
pub struct VulkanMemory {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the memory
	memory: VkDeviceMemory,

	/// The allocated size of the memory
	size: VkDeviceSize,

	/// The mapped address
	mapped_address: *mut c_void,

	/// The map counter
	pub(crate) map_count: Mutex<u32>,
}

/// The direction of manipulating data
#[derive(Debug)]
pub enum DataDirection {
	SetData,
	GetData
}

impl VulkanMemory {
	/// Create the `VulkanMemory`
	pub fn new(device: Arc<VulkanDevice>, mem_reqs: &VkMemoryRequirements, flags: VkMemoryPropertyFlags) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let alloc_i = VkMemoryAllocateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			pNext: null(),
			allocationSize: mem_reqs.size,
			memoryTypeIndex: device.get_gpu().get_memory_type_index(mem_reqs.memoryTypeBits, flags)?,
		};
		let mut memory: VkDeviceMemory = null();
		vkcore.vkAllocateMemory(device.get_vk_device(), &alloc_i, null(), &mut memory)?;
		let ret = Self {
			device,
			memory,
			size: mem_reqs.size,
			mapped_address: null_mut(),
			map_count: Mutex::new(0),
		};
		Ok(ret)
	}

	/// Get the `VkDeviceMemory`
	pub(crate) fn get_vk_memory(&self) -> VkDeviceMemory {
		self.memory
	}

	/// Get the length of the memory
	pub fn get_size(&self) -> VkDeviceSize {
		self.size
	}

	/// Map the memory
	pub fn map<'a>(&'a mut self, offset: VkDeviceSize, size: usize) -> Result<MappedMemory<'a>, VulkanError> {
		let mut map_count_lock = self.map_count.lock().unwrap();
		if *map_count_lock == 0 {
			*map_count_lock += 1;
			self.device.vkcore.vkMapMemory(self.device.get_vk_device(), self.memory, 0, self.size, 0, &mut self.mapped_address)?;
		}
		Ok(MappedMemory::new(self, (self.mapped_address as *mut u8).wrapping_add(offset as usize) as *mut c_void, size))
	}

	/// Map the memory as a slice
	pub fn map_as_slice<'a, T>(&'a mut self, offset: VkDeviceSize, size: usize) -> Result<TypedMappedMemory<'a, T>, VulkanError>
	where
		T: Sized + Clone + Copy {
		let mut map_count_lock = self.map_count.lock().unwrap();
		if *map_count_lock == 0 {
			*map_count_lock += 1;
			self.device.vkcore.vkMapMemory(self.device.get_vk_device(), self.memory, 0, self.size, 0, &mut self.mapped_address)?;
		}
		Ok(TypedMappedMemory::new(MappedMemory::new(self, (self.mapped_address as *mut u8).wrapping_add(offset as usize) as *mut c_void, size)))
	}

	/// Provide data for the memory, or retrieve data from the memory
	pub fn manipulate_data(&mut self, data: *mut c_void, offset: VkDeviceSize, size: usize, direction: DataDirection) -> Result<(), VulkanError> {
		let map_guard = self.map(offset, size)?;
		match direction {
			DataDirection::SetData => unsafe {copy(data as *const u8, map_guard.address as *mut u8, size)},
			DataDirection::GetData => unsafe {copy(map_guard.address as *const u8, data as *mut u8, size)},
		}
		Ok(())
	}

	/// Provide data for the memory
	pub fn set_data(&mut self, data: *const c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		self.manipulate_data(data as *mut c_void, offset, size, DataDirection::SetData)
	}

	/// Retrieve data from the memory
	pub fn get_data(&mut self, data: *mut c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		self.manipulate_data(data, offset, size, DataDirection::GetData)
	}

	/// Bind to a buffer
	pub(crate) fn bind_vk_buffer(&self, buffer: VkBuffer) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkBindBufferMemory(self.device.get_vk_device(), buffer, self.memory, 0)?;
		Ok(())
	}

	/// Bind to a image
	pub(crate) fn bind_vk_image(&self, image: VkImage) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkBindImageMemory(self.device.get_vk_device(), image, self.memory, 0)?;
		Ok(())
	}
}

impl Debug for VulkanMemory {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanMemory")
		.field("memory", &self.memory)
		.field("size", &self.size)
		.finish()
	}
}

impl Drop for VulkanMemory {
	fn drop(&mut self) {
		self.device.vkcore.vkFreeMemory(self.device.get_vk_device(), self.memory, null()).unwrap();
	}
}

/// The state that indicates the Vulkan memory is currently mapped
#[derive(Debug)]
pub struct MappedMemory<'a> {
	/// The reference to the memory
	pub memory: &'a VulkanMemory,

	/// The mapped address
	pub(crate) address: *mut c_void,

	/// The size of the map
	pub(crate) size: usize,
}

impl<'a> MappedMemory<'a> {
	/// Called by `VulkanMemory::map()`
	pub(crate) fn new(memory: &'a VulkanMemory, address: *mut c_void, size: usize) -> Self {
		Self {
			memory,
			address,
			size,
		}
	}

	/// Get the mapped address
	pub fn get_address(&self) -> *const c_void {
		self.address
	}

	/// Get the mapped size
	pub fn get_size(&self) -> usize {
		self.size
	}
}

impl Drop for MappedMemory<'_> {
	fn drop(&mut self) {
		let mut map_count_lock = self.memory.map_count.lock().unwrap();
		*map_count_lock -= 1;
		if *map_count_lock == 0 {
			self.memory.device.vkcore.vkUnmapMemory(self.memory.device.get_vk_device(), self.memory.memory).unwrap();
		}
	}
}

/// The typed map
#[derive(Debug)]
pub struct TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	/// The mapped memory
	mapped_memory: MappedMemory<'a>,

	/// The slice of items
	slice: &'a mut [T],

	/// The phantom data to hold the generic type T
	_phantom: PhantomData<T>,
}

impl<'a, T> TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	pub fn new(mapped_memory: MappedMemory<'a>) -> Self {
		let len = mapped_memory.size / size_of::<T>();
		let slice = unsafe {from_raw_parts_mut(mapped_memory.address as *mut T, len)};
		Self {
			mapped_memory,
			slice,
			_phantom: PhantomData,
		}
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

impl<'a, T> Index<usize> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = T;
	fn index(&self, index: usize) -> &T {
		&self.slice[index]
	}
}

impl<'a, T> IndexMut<usize> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, index: usize) -> &mut T {
		&mut self.slice[index]
	}
}

impl<'a, T> Index<Range<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: Range<usize>) -> &[T] {
		&self.slice[range.start..range.end]
	}
}

impl<'a, T> IndexMut<Range<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: Range<usize>) -> &mut [T] {
		&mut self.slice[range.start..range.end]
	}
}

impl<'a, T> Index<RangeFrom<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: RangeFrom<usize>) -> &[T] {
		&self.slice[range.start..]
	}
}

impl<'a, T> IndexMut<RangeFrom<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut [T] {
		&mut self.slice[range.start..]
	}
}

impl<'a, T> Index<RangeTo<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: RangeTo<usize>) -> &[T] {
		&self.slice[..range.end]
	}
}

impl<'a, T> IndexMut<RangeTo<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: RangeTo<usize>) -> &mut [T] {
		&mut self.slice[..range.end]
	}
}

impl<'a, T> Index<RangeFull> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, _: RangeFull) -> &[T] {
		&self.slice[..]
	}
}

impl<'a, T> IndexMut<RangeFull> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, _: RangeFull) -> &mut [T] {
		&mut self.slice[..]
	}
}

impl<'a, T> Index<RangeInclusive<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: RangeInclusive<usize>) -> &[T] {
		&self.slice[*range.start()..=*range.end()]
	}
}

impl<'a, T> IndexMut<RangeInclusive<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: RangeInclusive<usize>) -> &mut [T] {
		&mut self.slice[*range.start()..=*range.end()]
	}
}

impl<'a, T> Index<RangeToInclusive<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	type Output = [T];
	fn index(&self, range: RangeToInclusive<usize>) -> &[T] {
		&self.slice[..=range.end]
	}
}

impl<'a, T> IndexMut<RangeToInclusive<usize>> for TypedMappedMemory<'a, T>
where
	T: Sized + Clone + Copy {
	fn index_mut(&mut self, range: RangeToInclusive<usize>) -> &mut [T] {
		&mut self.slice[..=range.end]
	}
}

/// The buffer view range
#[derive(Debug, Clone, Copy)]
pub struct BufferViewRange {
	/// The format of the buffer view
	pub format: VkFormat,

	/// The offset of the view
	pub offset: VkDeviceSize,

	/// The range of the view
	pub range: VkDeviceSize,
}

/// The buffer view object for a buffer
pub struct VulkanBufferView {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The buffer
	buffer: Arc<VulkanBuffer>,

	/// The `VkBufferView`
	buffer_view: VkBufferView,

	/// The range of the buffer view
	range: BufferViewRange,
}

impl VulkanBufferView {
	/// Create the `VulkanBufferView` with a specific offset and range
	pub fn new_partial(buffer: Arc<VulkanBuffer>, range: &BufferViewRange) -> Result<Self, VulkanError> {
		let buffer_view_ci = VkBufferViewCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
			pNext: null(),
			flags: 0,
			buffer: buffer.get_vk_buffer(),
			format: range.format,
			offset: range.offset,
			range: range.range,
		};
		let mut buffer_view = null();
		buffer.device.vkcore.vkCreateBufferView(buffer.device.get_vk_device(), &buffer_view_ci, null(), &mut buffer_view)?;
		Ok(Self {
			device: buffer.device.clone(),
			buffer: buffer.clone(),
			buffer_view,
			range: *range,
		})
	}

	/// Create the `VulkanBufferView` that covers the whole buffer
	pub fn new(buffer: Arc<VulkanBuffer>, format: VkFormat) -> Result<Self, VulkanError> {
		let range = BufferViewRange {
			format,
			offset: 0,
			range: buffer.get_size(),
		};
		Self::new_partial(buffer, &range)
	}

	/// Get the `VkBufferView` handle
	pub(crate) fn get_vk_buffer_view(&self) -> VkBufferView {
		self.buffer_view
	}

	/// Get the range of the view
	pub fn get_range(&self) -> &BufferViewRange {
		&self.range
	}
}

impl Debug for VulkanBufferView {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanBufferView")
		.field("buffer", &self.buffer)
		.field("buffer_view", &self.buffer_view)
		.field("range", &self.range)
		.finish()
	}
}

impl Drop for VulkanBufferView {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyBufferView(self.device.get_vk_device(), self.buffer_view, null()).unwrap();
	}
}

/// The buffer object that temporarily stores the `VkBuffer`
pub struct VulkanBuffer {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The size of the buffer
	size: VkDeviceSize,

	/// The handle to the buffer
	buffer: VkBuffer,
}

impl VulkanBuffer {
	/// Create the `VulkanBuffer`
	pub fn new(device: Arc<VulkanDevice>, size: VkDeviceSize, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vkdevice = device.get_vk_device();
		let buffer_ci = VkBufferCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			pNext: null(),
			flags: 0,
			size,
			usage,
			sharingMode: VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
			queueFamilyIndexCount: 0,
			pQueueFamilyIndices: null(),
		};
		let mut buffer: VkBuffer = null();
		vkcore.vkCreateBuffer(vkdevice, &buffer_ci, null(), &mut buffer)?;
		Ok(Self {
			device,
			size,
			buffer,
		})
	}

	/// Get the memory requirements
	pub fn get_memory_requirements(&self) -> Result<VkMemoryRequirements, VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let mut ret: VkMemoryRequirements = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetBufferMemoryRequirements(self.device.get_vk_device(), self.buffer, &mut ret)?;
		Ok(ret)
	}

	/// Get the `VkBuffer` handle
	pub(crate) fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer
	}

	/// Get the size of the buffer
	pub fn get_size(&self) -> VkDeviceSize {
		self.size
	}
}

impl Debug for VulkanBuffer {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanBuffer")
		.field("size", &self.size)
		.field("buffer", &self.buffer)
		.finish()
	}
}

impl Drop for VulkanBuffer {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyBuffer(self.device.get_vk_device(), self.buffer, null()).unwrap();
	}
}

unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

/// The region of a buffer
#[derive(Debug)]
pub struct BufferRegion {
	pub offset: VkDeviceSize,
	pub size: VkDeviceSize,
}

/// The staging buffer for the Vulkan buffer object
pub struct StagingBuffer {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The device memory
	pub memory: VulkanMemory,

	/// The buffer
	pub buffer: Arc<VulkanBuffer>,

	/// The address of the data
	pub(crate) address: *mut c_void,
}

impl StagingBuffer {
	/// Create a new staging buffer
	pub fn new(device: Arc<VulkanDevice>, size: VkDeviceSize) -> Result<Self, VulkanError> {
		let buffer = Arc::new(VulkanBuffer::new(device.clone(), size, VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT as VkBufferUsageFlags)?);
		let memory = VulkanMemory::new(device.clone(), &buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT as VkMemoryPropertyFlags |
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT as VkMemoryPropertyFlags)?;
		memory.bind_vk_buffer(buffer.get_vk_buffer())?;
		let mut address: *mut c_void = null_mut();
		device.vkcore.vkMapMemory(device.get_vk_device(), memory.get_vk_memory(), 0, size, 0, &mut address)?;
		Ok(Self {
			device,
			memory,
			buffer,
			address,
		})
	}

	/// Get the `VkBuffer`
	pub(crate) fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.get_vk_buffer()
	}

	/// Get the `VkDeviceMemory`
	pub(crate) fn get_vk_memory(&self) -> VkDeviceMemory {
		self.memory.get_vk_memory()
	}

	/// Get the size of the staging buffer
	pub fn get_size(&self) -> VkDeviceSize {
		self.memory.get_size()
	}

	/// Get the address of the memory data
	pub fn get_address(&self) -> *mut c_void {
		self.address
	}

	/// Set the content of the staging buffer
	pub fn set_data(&mut self, data: *const c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		self.memory.set_data(data, offset, size)?;
		Ok(())
	}

	/// Set the content of the staging buffer
	pub fn get_data(&mut self, data: *mut c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		self.memory.get_data(data, offset, size)?;
		Ok(())
	}

	/// Map the memory
	pub fn map<'a>(&'a mut self, offset: VkDeviceSize, size: usize) -> Result<MappedMemory<'a>, VulkanError> {
		self.memory.map(offset, size)
	}
}

impl Debug for StagingBuffer {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("StagingBuffer")
		.field("memory", &self.memory)
		.field("buffer", &self.buffer)
		.field("address", &self.address)
		.finish()
	}
}

impl Drop for StagingBuffer {
	fn drop(&mut self) {
		self.device.vkcore.vkUnmapMemory(self.device.get_vk_device(), self.get_vk_memory()).unwrap();
	}
}

pub struct VulkanSampler {
	/// The device holds all of the resource
	pub device: Arc<VulkanDevice>,

	/// The sampler
	sampler: VkSampler,
}

impl VulkanSampler {
	/// Create the `VulkanSampler`
	pub fn new(device: Arc<VulkanDevice>, sampler_ci: &VkSamplerCreateInfo) -> Result<Self, VulkanError> {
		let mut sampler = null();
		device.vkcore.vkCreateSampler(device.get_vk_device(), sampler_ci, null(), &mut sampler)?;
		Ok(Self {
			device,
			sampler,
		})
	}

	/// Create the sampler that's most common used: filter = linear, address mode = repeat, mipmap = nearest
	pub fn new_linear(device: Arc<VulkanDevice>, with_mipmaps: bool, anisotropy: bool) -> Result<Self, VulkanError> {
		let max_anisotropy = device.get_gpu().properties.limits.maxSamplerAnisotropy;
		let sampler_ci = VkSamplerCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			pNext: null(),
			flags: 0,
			magFilter: VkFilter::VK_FILTER_LINEAR,
			minFilter: VkFilter::VK_FILTER_LINEAR,
			mipmapMode: VkSamplerMipmapMode::VK_SAMPLER_MIPMAP_MODE_NEAREST,
			addressModeU: VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT,
			addressModeV: VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT,
			addressModeW: VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT,
			mipLodBias: 0.0,
			anisotropyEnable: if anisotropy {1} else {0},
			maxAnisotropy: max_anisotropy,
			compareEnable: 0,
			compareOp: VkCompareOp::VK_COMPARE_OP_NEVER,
			minLod: 0.0,
			maxLod: if with_mipmaps {VK_LOD_CLAMP_NONE} else {0.0},
			borderColor: VkBorderColor::VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			unnormalizedCoordinates: 0,
		};
		Self::new(device, &sampler_ci)
	}

	/// Create the sampler that doesn't do interpolation between pixels
	pub fn new_nearest(device: Arc<VulkanDevice>, with_mipmaps: bool) -> Result<Self, VulkanError> {
		let sampler_ci = VkSamplerCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			pNext: null(),
			flags: 0,
			magFilter: VkFilter::VK_FILTER_NEAREST,
			minFilter: VkFilter::VK_FILTER_NEAREST,
			mipmapMode: VkSamplerMipmapMode::VK_SAMPLER_MIPMAP_MODE_NEAREST,
			addressModeU: VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT,
			addressModeV: VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT,
			addressModeW: VkSamplerAddressMode::VK_SAMPLER_ADDRESS_MODE_REPEAT,
			mipLodBias: 0.0,
			anisotropyEnable: 0,
			maxAnisotropy: 1.0,
			compareEnable: 0,
			compareOp: VkCompareOp::VK_COMPARE_OP_NEVER,
			minLod: 0.0,
			maxLod: if with_mipmaps {VK_LOD_CLAMP_NONE} else {0.0},
			borderColor: VkBorderColor::VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			unnormalizedCoordinates: 0,
		};
		Self::new(device, &sampler_ci)
	}

	/// Get the `VkSampler`
	pub fn get_vk_sampler(&self) -> VkSampler {
		self.sampler
	}
}

impl Debug for VulkanSampler {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanSampler")
		.field("sampler", &self.sampler)
		.finish()
	}
}

impl Drop for VulkanSampler {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroySampler(self.device.get_vk_device(), self.sampler, null()).unwrap();
	}
}
