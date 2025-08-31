
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	ptr::null,
	mem::MaybeUninit,
	sync::Arc,
};

/// The buffer object that temporarily stores the `VkBuffer`
struct Buffer {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the buffer
	buffer: VkBuffer,
}

impl Buffer {
	/// Create the `Buffer`
	fn new(device: Arc<VulkanDevice>, size: usize, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vkdevice = device.get_vk_device();
		let buffer_ci = VkBufferCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			pNext: null(),
			flags: 0,
			size: size as VkDeviceSize,
			usage,
			sharingMode: VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
			queueFamilyIndexCount: 0,
			pQueueFamilyIndices: null(),
		};
		let mut buffer: VkBuffer = null();
		vkcore.vkCreateBuffer(vkdevice, &buffer_ci, null(), &mut buffer)?;
		Ok(Self {
			device,
			buffer,
		})
	}

	fn get_memory_requirements(&self) -> Result<VkMemoryRequirements, VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let mut ret: VkMemoryRequirements = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetBufferMemoryRequirements(self.device.get_vk_device(), self.buffer, &mut ret)?;
		Ok(ret)
	}

	fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer
	}
}

impl Debug for Buffer {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("Buffer")
		.field("buffer", &self.buffer)
		.finish()
	}
}

impl Drop for Buffer {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkDestroyBuffer(self.device.get_vk_device(), self.buffer, null()).unwrap();
	}
}

/// The Vulkan buffer object, same as the OpenGL buffer object, could be used to store vertices, elements(indices), and the other data.
pub struct VulkanBuffer {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The device memory
	memory: VulkanMemory,

	/// The buffer
	buffer: Buffer,
}

impl VulkanBuffer {
	pub fn new(device: Arc<VulkanDevice>, size: usize, data: *const c_void, usage: VkBufferUsageFlags, queue_index: usize) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let staging_buffer = Buffer::new(device.clone(), size, VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT as u32)?;
		let staging_memory = VulkanMemory::new(device.clone(), &staging_buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT as u32 |
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT as u32)?;
		staging_memory.bind_buffer(staging_buffer.get_vk_buffer())?;
		staging_memory.set_data(data)?;
		let buffer = Buffer::new(device.clone(), size, usage | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT as u32)?;
		let memory = VulkanMemory::new(device.clone(), &buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as u32)?;
		memory.bind_buffer(buffer.get_vk_buffer())?;
		let mut command_pool = VulkanCommandPool::new(device.clone(), 1)?;
		let submit_fence = VulkanFence::new(device.clone())?;
		let pool_in_use = command_pool.use_pool(queue_index, None, true, Some(&submit_fence))?;
		let copy_region = VkBufferCopy {
			srcOffset: 0,
			dstOffset: 0,
			size: size as VkDeviceSize,
		};
		vkcore.vkCmdCopyBuffer(pool_in_use.cmdbuf, staging_buffer.buffer, buffer.buffer, 1, &copy_region)?;
		drop(pool_in_use);
		submit_fence.wait(u64::MAX)?;
		Ok(Self {
			device,
			memory,
			buffer,
		})
	}

	/// Get the buffer memory
	pub(crate) fn get_vk_memory(&self) -> VkDeviceMemory {
		self.memory.get_vk_memory()
	}

	/// Get the buffer
	pub(crate) fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.buffer
	}
}

impl Debug for VulkanBuffer {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanBuffer")
		.field("memory", &self.memory)
		.field("buffer", &self.buffer)
		.finish()
	}
}
