
use crate::prelude::*;
use std::{
	ffi::c_void,
	ptr::null,
	mem::MaybeUninit,
	sync::{Arc, Mutex},
};

/// The buffer object that temporarily stores the `VkBuffer`
#[derive(Debug)]
struct Buffer {
	/// The `VulkanContext` that helps to manage the resources of the buffer
	ctx: Arc<Mutex<VulkanContext>>,

	/// The handle to the buffer
	buffer: VkBuffer,
}

impl Buffer {
	/// Create the `Buffer`
	fn new(ctx: Arc<Mutex<VulkanContext>>, size: usize, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let lock = ctx.lock().unwrap();
		let vkcore = lock.get_vkcore();
		let vkdevice = lock.get_vk_device();
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
		drop(lock);
		Ok(Self {
			ctx,
			buffer,
		})
	}

	fn get_memory_requirements(&self) -> Result<VkMemoryRequirements, VulkanError> {
		let lock = self.ctx.lock().unwrap();
		let vkcore = lock.get_vkcore();
		let vkdevice = lock.get_vk_device();
		let mut ret: VkMemoryRequirements = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetBufferMemoryRequirements(vkdevice, self.buffer, &mut ret)?;
		Ok(ret)
	}

	fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer
	}
}

impl Drop for Buffer {
	fn drop(&mut self) {
		let lock = self.ctx.lock().unwrap();
		let vkcore = lock.get_vkcore();
		let vkdevice = lock.get_vk_device();
		vkcore.vkDestroyBuffer(vkdevice, self.buffer, null()).unwrap();
	}
}

/// The Vulkan buffer object, same as the OpenGL buffer object, could be used to store vertices, elements(indices), and the other data.
#[derive(Debug)]
pub struct VulkanBuffer {
	/// The `VulkanContext` that helps to manage the resources of the buffer
	ctx: Arc<Mutex<VulkanContext>>,

	/// The device memory
	memory: VulkanMemory,

	/// The buffer
	buffer: Buffer,
}

impl VulkanBuffer {
	pub fn new(ctx: Arc<Mutex<VulkanContext>>, size: usize, data: *const c_void, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let staging_buffer = Buffer::new(ctx.clone(), size, VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT as u32)?;
		let staging_memory = VulkanMemory::new(ctx.clone(), &staging_buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT as u32 |
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT as u32)?;
		staging_memory.bind_buffer(staging_buffer.get_vk_buffer())?;
		staging_memory.set_data(data)?;
		let buffer = Buffer::new(ctx.clone(), size, usage | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT as u32)?;
		let memory = VulkanMemory::new(ctx.clone(), &buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as u32)?;
		memory.bind_buffer(buffer.get_vk_buffer())?;
		let mut command_pool = VulkanCommandPool::new(ctx.clone(), 1)?;
		let submit_fence = VulkanFence::new(ctx.clone())?;
		let pool_in_use = command_pool.use_pool(None, None, true, Some(&submit_fence))?;
		let copy_region = VkBufferCopy {
			srcOffset: 0,
			dstOffset: 0,
			size: size as VkDeviceSize,
		};
		pool_in_use.vkcore.vkCmdCopyBuffer(pool_in_use.cmdbuf, staging_buffer.buffer, buffer.buffer, 1, &copy_region)?;
		drop(pool_in_use);
		submit_fence.wait(u64::MAX)?;
		Ok(Self {
			ctx,
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
