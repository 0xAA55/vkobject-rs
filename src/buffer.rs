
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

	/// The staging buffer
	pub staging_buffer: Option<StagingBuffer>,
}

impl Buffer {
	/// Create a new buffer
	/// * If `data` is `None`, `cmdbuf` could be `null()` because no `vkCmdCopyBuffer()` will be issued.
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
			staging_buffer: None,
		};
		if let Some(data) = data {
			ret.set_staging_data(data, 0, size as usize)?;
		}
		Ok(ret)
	}

	/// Update new data to the buffer
	pub fn set_staging_data(&mut self, data: *const c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		if let None = self.staging_buffer {
			self.staging_buffer = Some(StagingBuffer::new(self.device.clone(), self.memory.get_size())?);
		}
		self.staging_buffer.as_ref().unwrap().set_data(data, offset, size)?;
		Ok(())
	}

	/// Upload the data from the staging buffer
	pub fn upload_staging_buffer(&self, cmdbuf: VkCommandBuffer, offset: VkDeviceSize, size: VkDeviceSize) -> Result<(), VulkanError> {
		let copy_region = VkBufferCopy {
			srcOffset: offset,
			dstOffset: offset,
			size: size as VkDeviceSize,
		};
		self.device.vkcore.vkCmdCopyBuffer(cmdbuf, self.staging_buffer.as_ref().unwrap().get_vk_buffer(), self.buffer.get_vk_buffer(), 1, &copy_region)?;
		Ok(())
	}

	/// Upload the data from the staging buffer
	pub fn upload_staging_buffer_multi(&self, cmdbuf: VkCommandBuffer, regions: &[BufferRegion]) -> Result<(), VulkanError> {
		let copy_regions: Vec<VkBufferCopy> = regions.iter().map(|r|VkBufferCopy {
			srcOffset: r.offset,
			dstOffset: r.offset,
			size: r.size as VkDeviceSize,
		}).collect();
		self.device.vkcore.vkCmdCopyBuffer(cmdbuf, self.staging_buffer.as_ref().unwrap().get_vk_buffer(), self.buffer.get_vk_buffer(), copy_regions.len() as u32, copy_regions.as_ptr())?;
		Ok(())
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&mut self) {
		self.staging_buffer = None;
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
