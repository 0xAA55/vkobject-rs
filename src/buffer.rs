
use crate::prelude::*;
use std::{
	ffi::c_void,
	ptr::{null, null_mut, copy},
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
}

impl Drop for Buffer {
	fn drop(&mut self) {
		let lock = self.ctx.lock().unwrap();
		let vkcore = lock.get_vkcore();
		let vkdevice = lock.get_vk_device();
		vkcore.vkDestroyBuffer(vkdevice, self.buffer, null()).unwrap();
	}
}

	/// Get the buffer memory
	pub fn get_memory(&self) -> VkDeviceMemory {
		self.memory
	}

	/// Get the buffer
	pub fn get_buffer(&self) -> VkBuffer {
		self.buffer
	}
}
