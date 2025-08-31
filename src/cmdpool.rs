
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	ptr::null,
	sync::{Arc, Mutex, MutexGuard},
};

/// The Vulkan command pool, and the associated buffers, fence. Support multiple buffers; you can use one buffer for command recording and another for submitting to a queue, interleaved.
pub struct VulkanCommandPool {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the command pool
	pool: Mutex<VkCommandPool>,

	/// The command buffers of the command pool
	cmd_buffers: Vec<VkCommandBuffer>,

	/// The last command buffer index
	pub last_buf_index: Mutex<u32>,

	/// The fence for the command pool
	pub(crate) fence: VulkanFence,
}

unsafe impl Send for VulkanCommandPool {}

impl VulkanCommandPool {
	/// Create a new `VulkanCommandPool`
	pub fn new(device: Arc<VulkanDevice>, num_buffers: usize) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vk_device = device.get_vk_device();
		let pool_ci = VkCommandPoolCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			pNext: null(),
			queueFamilyIndex: device.get_queue_family_index(),
			flags: VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT as u32,
		};
		let mut pool: VkCommandPool = null();
		vkcore.vkCreateCommandPool(vk_device, &pool_ci, null(), &mut pool)?;
		let cmd_buffers_ci = VkCommandBufferAllocateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			pNext: null(),
			commandPool: pool,
			level: VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			commandBufferCount: num_buffers as u32,
		};
		let mut cmd_buffers: Vec<VkCommandBuffer> = Vec::with_capacity(num_buffers);
		vkcore.vkAllocateCommandBuffers(vk_device, &cmd_buffers_ci, cmd_buffers.as_mut_ptr())?;
		unsafe {cmd_buffers.set_len(num_buffers)};
		let fence = VulkanFence::new(device.clone())?;
		Ok(Self{
			device,
			pool: Mutex::new(pool),
			cmd_buffers,
			last_buf_index: Mutex::new(0),
			fence,
		})
	}

	/// Retrieve the command pool
	pub(crate) fn get_vk_cmdpool<'a>(&'a self) -> MutexGuard<'a, VkCommandPool> {
		self.pool.lock().unwrap()
	}

	/// Retrieve the command pool
	pub(crate) fn try_get_vk_cmdpool<'a>(&'a self) -> Result<MutexGuard<'a, VkCommandPool>, VulkanError> {
		match self.pool.try_lock() {
			Ok(guard) => Ok(guard),
			_ => Err(VulkanError::CommandPoolIsInUse),
		}
	}

	/// Get the command buffers
	pub(crate) fn get_vk_cmd_buffers(&self) -> &[VkCommandBuffer] {
		&self.cmd_buffers
	}

	/// Get the fences
	pub(crate) fn get_vk_fence(&self) -> VkFence {
		self.fence.get_vk_fence()
	}

	/// Update the buffer index
	fn get_next_vk_cmd_buffer(&mut self) -> VkCommandBuffer {
		let mut lock = self.last_buf_index.lock().unwrap();
		let cmdbuf_index = *lock as usize;
		*lock += 1;
		if *lock as usize >= self.cmd_buffers.len() {
			*lock = 0;
		}
		self.cmd_buffers[cmdbuf_index]
	}

	/// Use a command buffer of the command pool to record draw commands
	pub(crate) fn use_pool(&mut self, queue_index: usize, swapchain_image: Option<Arc<Mutex<VulkanSwapchainImage>>>, one_time_submit: bool, submit_fence: Option<&VulkanFence>) -> Result<VulkanCommandPoolInUse, VulkanError> {
		let pool = *self.get_vk_cmdpool();
		let buf = self.get_next_vk_cmd_buffer();
		let submit_fence = if let Some(submit_fence) = submit_fence {
			submit_fence.get_vk_fence()
		} else {
			self.fence.get_vk_fence()
		};
		VulkanCommandPoolInUse::new(self, pool, buf, queue_index, swapchain_image, one_time_submit, submit_fence)
	}

	/// Try to acquire the command pool to record draw commands
	pub(crate) fn try_use_pool(&mut self, queue_index: usize, swapchain_image: Option<Arc<Mutex<VulkanSwapchainImage>>>, one_time_submit: bool, submit_fence: Option<&VulkanFence>) -> Result<VulkanCommandPoolInUse, VulkanError> {
		let pool = *self.try_get_vk_cmdpool()?;
		let buf = self.get_next_vk_cmd_buffer();
		let submit_fence = if let Some(submit_fence) = submit_fence {
			submit_fence.get_vk_fence()
		} else {
			self.fence.get_vk_fence()
		};
		VulkanCommandPoolInUse::new(self, pool, buf, queue_index, swapchain_image, one_time_submit, submit_fence)
	}
}

impl Debug for VulkanCommandPool {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanCommandPool")
		.field("pool", &self.pool)
		.field("cmd_buffers", &self.cmd_buffers)
		.field("last_buf_index", &self.last_buf_index)
		.field("fence", &self.fence)
		.finish()
	}
}

impl Drop for VulkanCommandPool {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkDestroyCommandPool(self.device.get_vk_device(), *self.pool.lock().unwrap(), null()).unwrap();
	}
}

/// The RAII wrapper for the usage of a Vulkan command pool/buffer. When created, your command could be recorded to the command buffer.
pub struct VulkanCommandPoolInUse {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The command buffer we are using here
	pub(crate) cmdbuf: VkCommandBuffer,

	/// The command pool to submit commands
	pub(crate) pool: VkCommandPool,

	/// The queue index for the command pool to submit
	pub(crate) queue_index: usize,

	/// The swapchain image index for the command pool to draw to
	pub swapchain_image: Option<Arc<Mutex<VulkanSwapchainImage>>>,

	/// The fence indicating if all commands were submitted
	pub(crate) submit_fence: VkFence,

	/// Is this command buffer got automatically cleaned when submitted
	pub(crate) one_time_submit: bool,

	/// Is recording commands ended
	pub(crate) ended: bool,

	/// Is the commands submitted
	pub(crate) submitted: bool,
}

impl VulkanCommandPoolInUse {
	/// Create a RAII binding to the `VulkanCommandPool` in use
	fn new(cmdpool: &VulkanCommandPool, pool: VkCommandPool, cmdbuf: VkCommandBuffer, queue_index: usize, swapchain_image: Option<Arc<Mutex<VulkanSwapchainImage>>>, one_time_submit: bool, submit_fence: VkFence) -> Result<Self, VulkanError> {
		let vkcore = cmdpool.device.vkcore.clone();
		let device = cmdpool.device.clone();
		let begin_info = VkCommandBufferBeginInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			pNext: null(),
			flags: if one_time_submit {VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT as u32} else {0u32},
			pInheritanceInfo: null(),
		};
		vkcore.vkBeginCommandBuffer(cmdbuf, &begin_info)?;
		Ok(Self {
			device,
			cmdbuf,
			pool,
			queue_index,
			swapchain_image,
			submit_fence,
			one_time_submit,
			ended: false,
			submitted: false,
		})
	}

	/// Get the current command buffer
	pub(crate) fn get_vk_cmdbuf(&self) -> VkCommandBuffer {
		self.cmdbuf
	}

	/// Is this command buffer an one time submit command buffer
	pub fn is_one_time_submit(&self) -> bool {
		self.one_time_submit
	}

	/// End recording commands
	pub fn end_cmd(&mut self) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		if !self.ended {
			vkcore.vkEndCommandBuffer(self.get_vk_cmdbuf())?;
			self.ended = true;
			Ok(())
		} else {
			panic!("Duplicated call to `VulkanCommandPoolInUse::end()`")
		}
	}

	/// Check if is ended
	pub fn is_ended(&self) -> bool {
		self.ended
	}

	/// Submit the commands
	pub fn submit(&mut self) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		if !self.ended {
			self.end_cmd()?;
		}
		if !self.submitted {
			let wait_stage = [VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT as VkPipelineStageFlags];
			let cmd_buffers = [self.cmdbuf];
			let mut submit_info = VkSubmitInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_SUBMIT_INFO,
				pNext: null(),
				waitSemaphoreCount: 0,
				pWaitSemaphores: null(),
				pWaitDstStageMask: wait_stage.as_ptr(),
				commandBufferCount: 1,
				pCommandBuffers: cmd_buffers.as_ptr(),
				signalSemaphoreCount: 0,
				pSignalSemaphores: null(),
			};
			let mut acquire_semaphores: Vec<VkSemaphore> = Vec::new();
			let mut release_semaphores: Vec<VkSemaphore> = Vec::new();
			if let Some(swapchain_image) = &self.swapchain_image {
				let lock = swapchain_image.lock().unwrap();
				acquire_semaphores.push(lock.acquire_semaphore.get_vk_semaphore());
				release_semaphores.push(lock.release_semaphore.get_vk_semaphore());
				drop(lock);
				submit_info.waitSemaphoreCount = acquire_semaphores.len() as u32;
				submit_info.signalSemaphoreCount = release_semaphores.len() as u32;
				submit_info.pWaitSemaphores = acquire_semaphores.as_ptr();
				submit_info.pSignalSemaphores = release_semaphores.as_ptr();
			}
			let submits = [submit_info];
			vkcore.vkQueueSubmit(self.device.get_vk_queue(self.queue_index), submits.len() as u32, submits.as_ptr(), self.submit_fence)?;
			self.submitted = true;
			Ok(())
		} else {
			panic!("Duplicated call to `VulkanCommandPoolInUse::submit()`, please set the `submitted` member to false to re-submit again if you wish.")
		}
	}

	/// End recording to the command buffer and submit the commands to the queue
	pub fn end(self) {}
}

impl Debug for VulkanCommandPoolInUse {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanCommandPoolInUse")
		.field("cmdbuf", &self.cmdbuf)
		.field("pool", &self.pool)
		.field("queue_index", &self.queue_index)
		.field("swapchain_image", &self.swapchain_image)
		.field("submit_fence", &self.submit_fence)
		.field("one_time_submit", &self.one_time_submit)
		.field("ended", &self.ended)
		.field("submitted", &self.submitted)
		.finish()
	}
}

impl Drop for VulkanCommandPoolInUse {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		if !self.submitted {
			self.submit().unwrap();
		}
		if !self.one_time_submit {
			vkcore.vkResetCommandBuffer(self.cmdbuf, 0).unwrap();
		}
	}
}
