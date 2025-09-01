
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	sync::Arc,
};

/// The Vulkan buffer object, same as the OpenGL buffer object, could be used to store vertices, elements(indices), and the other data.
pub struct Buffer {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The device memory
	pub memory: VulkanMemory,

	/// The buffer
	pub buffer: VulkanBuffer,

	/// The usage of the buffer, not including `VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT` and `VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT`
	pub(crate) usage: VkBufferUsageFlags,
}

impl Buffer {
	/// Create a new buffer
	pub fn new(device: Arc<VulkanDevice>, size: VkDeviceSize, data: Option<*const c_void>, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let buffer = VulkanBuffer::new(device.clone(), size, usage | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT as u32)?;
		let memory = VulkanMemory::new(device.clone(), &buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as u32)?;
		memory.bind_vk_buffer(buffer.get_vk_buffer())?;
		let mut ret = Self {
			device,
			memory,
			buffer,
			usage,
		};
		if let Some(data) = data {
			ret.set_data(data, 0, size)?;
		}
		Ok(ret)
	}

	/// Update new data to the buffer
	pub fn set_data(&self, data: *const c_void, offset: VkDeviceSize, length: VkDeviceSize) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let staging_buffer = VulkanBuffer::new(self.device.clone(), length, VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT as u32)?;
		let staging_memory = VulkanMemory::new(self.device.clone(), &staging_buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT as u32 |
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT as u32)?;
		staging_memory.bind_vk_buffer(staging_buffer.get_vk_buffer())?;
		staging_memory.set_data(data)?;
		let mut command_pool = VulkanCommandPool::new(self.device.clone(), 1)?;
		let copy_region = VkBufferCopy {
			srcOffset: 0,
			dstOffset: offset,
			size: length as VkDeviceSize,
		};
		vkcore.vkCmdCopyBuffer(pool_in_use.cmdbuf, staging_buffer.get_vk_buffer(), self.buffer.get_vk_buffer(), 1, &copy_region)?;
		Ok(())
	}
}

impl Debug for Buffer {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("Buffer")
		.field("memory", &self.memory)
		.field("buffer", &self.buffer)
		.field("usage", &self.usage)
		.finish()
	}
}
