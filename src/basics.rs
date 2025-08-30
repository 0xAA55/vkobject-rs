
#![allow(clippy::uninit_vec)]
#![allow(clippy::too_many_arguments)]
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	ptr::{null, null_mut, copy},
	sync::Arc,
};

/// The error for almost all of the crate's `Result<>`
#[derive(Debug, Clone)]
pub enum VulkanError {
	VkError(VkError),
	ChooseGpuFailed,
	NoGoodQueueForSurface(&'static str),
	NoGoodDepthStencilFormat,
	CommandPoolIsInUse,
	NoIdleCommandPools,
	NoIdleDeviceQueues,
	NoSuitableMemoryType,
}

impl From<VkError> for VulkanError {
	fn from(e: VkError) -> Self {
		Self::VkError(e)
	}
}

/// The wrapper for the `VkSemaphore`
pub struct VulkanSemaphore {
	/// The `VulkanDevice` is the associated device
	device: Arc<VulkanDevice>,

	/// The semaphore handle
	semaphore: VkSemaphore,

	/// For the timeline semaphore, this is the timeline value
	pub(crate) timeline: u64,
}

unsafe impl Send for VulkanSemaphore {}

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
				flags: if any {VkSemaphoreWaitFlagBits::VK_SEMAPHORE_WAIT_ANY_BIT as u32} else {0},
				semaphoreCount: semaphores.len() as u32,
				pSemaphores: semaphores.as_ptr(),
				pValues: timelines.as_ptr(),
			};
			vkcore.vkWaitSemaphores(vk_device, &wait_i, timeout)?;
			Ok(())
		}
	}

	/// Wait for multiple semaphores
	pub fn wait_multi_vk(ctx: &VulkanContext, semaphores: &[VkSemaphore], timelines: &[u64], timeout: u64, any: bool) -> Result<(), VulkanError> {
		if semaphores.is_empty() {
			Ok(())
		} else {
			let vkcore = ctx.vkcore.clone();
			let vk_device = ctx.get_vk_device();
			let wait_i = VkSemaphoreWaitInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
				pNext: null(),
				flags: if any {VkSemaphoreWaitFlagBits::VK_SEMAPHORE_WAIT_ANY_BIT as u32} else {0},
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
	device: Arc<VulkanDevice>,

	/// The fence handle
	fence: VkFence,
}

unsafe impl Send for VulkanFence {}

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
	pub fn unsignal_multi_vk(ctx: &VulkanContext, fences: &[VkFence]) -> Result<(), VulkanError> {
		if fences.is_empty() {
			Ok(())
		} else {
			let vkcore = ctx.get_vkcore();
			Ok(vkcore.vkResetFences(ctx.get_vk_device(), fences.len() as u32, fences.as_ptr())?)
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
	pub fn wait_multi_vk(ctx: &VulkanContext, fences: &[VkFence], timeout: u64, any: bool) -> Result<(), VulkanError> {
		if fences.is_empty() {
			Ok(())
		} else {
			let vkcore = ctx.vkcore.clone();
			let vk_device = ctx.get_vk_device();
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
	device: Arc<VulkanDevice>,

	/// The handle to the memory
	memory: VkDeviceMemory,

	/// The allocated size of the memory
	size: VkDeviceSize,
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
		};
		Ok(ret)
	}

	/// Get the `VkDeviceMemory`
	pub(crate) fn get_vk_memory(&self) -> VkDeviceMemory {
		self.memory
	}

	/// Provide data for the memory, or retrieve data from the memory
	pub fn manipulate_data(&self, data: *mut c_void, direction: DataDirection) -> Result<(), VulkanError> {
		let vkdevice = self.device.get_vk_device();
		let mut map_pointer: *mut c_void = null_mut();
		self.device.vkcore.vkMapMemory(vkdevice, self.memory, 0, self.size, 0, &mut map_pointer)?;
		match direction {
			DataDirection::SetData => unsafe {copy(data as *const u8, map_pointer as *mut u8, self.size as usize)},
			DataDirection::GetData => unsafe {copy(map_pointer as *const u8, data as *mut u8, self.size as usize)},
		}
		self.device.vkcore.vkUnmapMemory(vkdevice, self.memory)?;
		Ok(())
	}

	/// Provide data for the memory
	pub fn set_data(&self, data: *const c_void) -> Result<(), VulkanError> {
		self.manipulate_data(data as *mut c_void, DataDirection::SetData)
	}

	/// Retrieve data from the memory
	pub fn get_data(&self, data: *mut c_void) -> Result<(), VulkanError> {
		self.manipulate_data(data, DataDirection::GetData)
	}

	/// Bind to a buffer
	pub fn bind_buffer(&self, buffer: VkBuffer) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkBindBufferMemory(self.device.get_vk_device(), buffer, self.memory, 0)?;
		Ok(())
	}

	/// Bind to a image
	pub fn bind_image(&self, image: VkImage) -> Result<(), VulkanError> {
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
		let vkcore = self.device.vkcore.clone();
		vkcore.vkFreeMemory(self.device.get_vk_device(), self.memory, null()).unwrap();
	}
}
