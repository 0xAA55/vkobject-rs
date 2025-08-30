
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

/// The memory object that temporarily stores the `VkDeviceMemory`
#[derive(Debug)]
struct Memory {
	/// The `VulkanContext` that helps to manage the resources of the buffer
	ctx: Arc<Mutex<VulkanContext>>,

	/// The handle to the memory
	memory: VkDeviceMemory,

	/// The allocated size of the memory
	size: VkDeviceSize,
}

/// The direction of manipulating data
#[derive(Debug)]
enum DataDirection {
	SetData,
	GetData
}

impl Memory {
	/// Create the `Memory`
	fn new(ctx: Arc<Mutex<VulkanContext>>, buffer: &Buffer, flags: VkMemoryPropertyFlags) -> Result<Self, VulkanError> {
		let lock = ctx.lock().unwrap();
		let vkcore = lock.vkcore.clone();
		let vkdevice = lock.get_vk_device();
		let mem_reqs = buffer.get_memory_requirements()?;
		let alloc_i = VkMemoryAllocateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			pNext: null(),
			allocationSize: mem_reqs.size,
			memoryTypeIndex: lock.device.get_gpu().get_memory_type_index(mem_reqs.memoryTypeBits, flags)?,
		};
		drop(lock);
		let mut memory: VkDeviceMemory = null();
		vkcore.vkAllocateMemory(vkdevice, &alloc_i, null(), &mut memory)?;
		let ret = Self {
			ctx,
			memory,
			size: mem_reqs.size,
		};
		vkcore.vkBindBufferMemory(vkdevice, buffer.buffer, memory, 0)?;
		Ok(ret)
	}

	/// Provide data for the memory, or retrieve data from the memory
	fn manipulate_data(&self, data: *mut c_void, direction: DataDirection) -> Result<(), VulkanError> {
		let lock = self.ctx.lock().unwrap();
		let vkcore = lock.get_vkcore();
		let vkdevice = lock.get_vk_device();
		let mut map_pointer: *mut c_void = null_mut();
		vkcore.vkMapMemory(vkdevice, self.memory, 0, self.size, 0, &mut map_pointer)?;
		match direction {
			DataDirection::SetData => unsafe {copy(data as *const u8, map_pointer as *mut u8, self.size as usize)},
			DataDirection::GetData => unsafe {copy(map_pointer as *const u8, data as *mut u8, self.size as usize)},
		}
		vkcore.vkUnmapMemory(vkdevice, self.memory)?;
		Ok(())
	}

	/// Provide data for the memory
	fn set_data(&self, data: *const c_void) -> Result<(), VulkanError> {
		self.manipulate_data(data as *mut c_void, DataDirection::SetData)
	}

	/// Retrieve data from the memory
	fn get_data(&self, data: *mut c_void) -> Result<(), VulkanError> {
		self.manipulate_data(data, DataDirection::GetData)
	}

	/// Bind to a buffer
	fn bind_buffer(&self, buffer: VkBuffer) -> Result<(), VulkanError> {
		let lock = self.ctx.lock().unwrap();
		let vkcore = lock.get_vkcore();
		let vkdevice = lock.get_vk_device();
		vkcore.vkBindBufferMemory(vkdevice, buffer, self.memory, 0)?;
		Ok(())
	}
}

impl Drop for Memory {
	fn drop(&mut self) {
		let lock = self.ctx.lock().unwrap();
		let vkcore = lock.get_vkcore();
		let vkdevice = lock.get_vk_device();
		vkcore.vkFreeMemory(vkdevice, self.memory, null()).unwrap();
	}
}

/// The Vulkan buffer object, same as the OpenGL buffer object, could be used to store vertices, elements(indices), and the other data.
#[derive(Debug)]
pub struct VulkanBuffer {
	/// The `VulkanContext` that helps to manage the resources of the buffer
	ctx: Arc<Mutex<VulkanContext>>,

	/// The device memory
	memory: Memory,

	/// The buffer
	buffer: Buffer,
}

impl VulkanBuffer {
	pub fn new(ctx: Arc<Mutex<VulkanContext>>, size: usize, data: *const c_void, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let staging_buffer = Buffer::new(ctx.clone(), size, VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT as u32)?;
		let staging_memory = Memory::new(ctx.clone(), &staging_buffer,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT as u32 |
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT as u32)?;
		staging_memory.set_data(data)?;
		let buffer = Buffer::new(ctx.clone(), size, usage | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT as u32)?;
		let memory = Memory::new(ctx.clone(), &buffer,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as u32)?;
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
		submit_fence.wait(u64::MAX);
		Ok(Self {
			ctx,
			memory,
			buffer,
		})
	}

	/// Get the buffer memory
	pub(crate) fn get_vk_memory(&self) -> VkDeviceMemory {
		self.memory.memory
	}

	/// Get the buffer
	pub(crate) fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.buffer
	}
}
