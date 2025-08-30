
use crate::prelude::*;
use std::{
	fmt::Debug,
	ptr::null,
	sync::{Arc, Mutex, MutexGuard, Weak},
};

/// The Vulkan command pool, and the associated buffers, fence. Support multiple buffers; you can use one buffer for command recording and another for submitting to a queue, interleaved.
#[derive(Debug)]
pub struct VulkanCommandPool {
	/// The `VulkanContext` that helps to manage the resources of the command pool
	ctx: Weak<Mutex<VulkanContext>>,

	/// The handle to the command pool
	pool: Mutex<VkCommandPool>,

	/// The command buffers of the command pool
	cmd_buffers: Vec<VkCommandBuffer>,

	/// The last command buffer index
	pub last_buf_index: u32,

	/// The fence for the command pool
	pub(crate) fence: VulkanFence,
}

unsafe impl Send for VulkanCommandPool {}

impl VulkanCommandPool {
	/// Create a new `VulkanCommandPool`
	pub fn new_(vkcore: &VkCore, device: &VulkanDevice, num_buffers: usize) -> Result<Self, VulkanError> {
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
		Ok(Self{
			ctx: Weak::new(),
			pool: Mutex::new(pool),
			cmd_buffers,
			last_buf_index: 0,
			fence: VulkanFence::new_(vkcore, vk_device)?,
		})
	}

	/// Create a new `VulkanCommandPool`
	pub fn new(ctx: Arc<Mutex<VulkanContext>>, num_buffers: usize) -> Result<Self, VulkanError> {
		let ctx_lock = ctx.lock().unwrap();
		let vkcore = ctx_lock.get_vkcore();
		let device = &ctx_lock.device;
		let mut ret = Self::new_(vkcore, device, num_buffers)?;
		ret.set_ctx(Arc::downgrade(&ctx));
		Ok(ret)
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

	/// Set the context
	pub(crate) fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.fence.set_ctx(ctx.clone());
		self.ctx = ctx;
	}

	/// Update the buffer index
	fn get_next_vk_cmd_buffer(&mut self) -> VkCommandBuffer {
		let cmdbuf_index = self.last_buf_index as usize;
		self.last_buf_index += 1;
		if self.last_buf_index as usize > self.cmd_buffers.len() {
			self.last_buf_index = 0;
		}
		self.cmd_buffers[cmdbuf_index]
	}

	/// Use a command buffer of the command pool to record draw commands
	pub fn use_pool<'a>(&'a mut self, queue_index: Option<usize>, swapchain_image_index: Option<usize>, one_time_submit: bool, submit_fence: Option<&VulkanFence>) -> Result<VulkanCommandPoolInUse<'a>, VulkanError> {
		let buf = self.get_next_vk_cmd_buffer();
		let submit_fence = if let Some(submit_fence) = submit_fence {
			submit_fence.get_vk_fence()
		} else {
			self.fence.get_vk_fence()
		};
		VulkanCommandPoolInUse::new(self, self.get_vk_cmdpool(), buf, queue_index, swapchain_image_index, one_time_submit, submit_fence)
	}

	/// Try to acquire the command pool to record draw commands
	pub fn try_use_pool<'a>(&'a mut self, queue_index: Option<usize>, swapchain_image_index: Option<usize>, one_time_submit: bool, submit_fence: Option<&VulkanFence>) -> Result<VulkanCommandPoolInUse<'a>, VulkanError> {
		let buf = self.get_next_vk_cmd_buffer();
		let submit_fence = if let Some(submit_fence) = submit_fence {
			submit_fence.get_vk_fence()
		} else {
			self.fence.get_vk_fence()
		};
		VulkanCommandPoolInUse::new(self, self.try_get_vk_cmdpool()?, buf, queue_index, swapchain_image_index, one_time_submit, submit_fence)
	}
}

impl Drop for VulkanCommandPool {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			vkcore.vkDestroyCommandPool(ctx.get_vk_device(), *self.pool.lock().unwrap(), null()).unwrap();
		}
	}
}

/// The RAII wrapper for the usage of a Vulkan command pool/buffer. When created, your command could be recorded to the command buffer.
#[derive(Debug)]
pub struct VulkanCommandPoolInUse<'a> {
	/// The `VulkanContext` that helps to manage the resources of the command pool
	pub(crate) ctx: Arc<Mutex<VulkanContext>>,

	/// The `VkCore` here for the convenience of calling Vulkan functions
	pub(crate) vkcore: Arc<VkCore>,

	/// The command buffer we are using here
	pub(crate) cmdbuf: VkCommandBuffer,

	/// The locked state of the `VkCommandPool`
	pool_lock: MutexGuard<'a, VkCommandPool>,

	/// The queue index for the command pool to submit
	pub(crate) queue_index: Option<usize>,

	/// The swapchain image index for the command pool to draw to
	pub(crate) swapchain_image_index: Option<usize>,

	/// The fence indicating if all commands were submitted
	pub(crate) submit_fence: VkFence,

	/// Is this command buffer got automatically cleaned when submitted
	pub(crate) one_time_submit: bool,

	/// Is recording commands ended
	pub(crate) ended: bool,

	/// Is the commands submitted
	pub(crate) submitted: bool,
}

impl<'a> VulkanCommandPoolInUse<'a> {
	/// Create a RAII binding to the `VulkanCommandPool` in use
	fn new(cmdpool: &VulkanCommandPool, pool_lock: MutexGuard<'a, VkCommandPool>, cmdbuf: VkCommandBuffer, queue_index: Option<usize>, swapchain_image_index: Option<usize>, one_time_submit: bool, submit_fence: VkFence) -> Result<Self, VulkanError> {
		let ctx = cmdpool.ctx.upgrade().unwrap();
		let ctx_g = ctx.lock().unwrap();
		let vkcore = ctx_g.vkcore.clone();
		let begin_info = VkCommandBufferBeginInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			pNext: null(),
			flags: if one_time_submit {VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT as u32} else {0u32},
			pInheritanceInfo: null(),
		};
		vkcore.vkBeginCommandBuffer(cmdbuf, &begin_info)?;
		Ok(Self {
			ctx: ctx.clone(),
			vkcore,
			cmdbuf,
			pool_lock,
			queue_index,
			swapchain_image_index,
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
		if !self.ended {
			let ctx = self.ctx.lock().unwrap();
			let vkcore = ctx.get_vkcore();
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
		if !self.ended {
			self.end_cmd()?;
		}
		if !self.submitted {
			let ctx = self.ctx.lock().unwrap();
			let vkcore = ctx.get_vkcore();

			let wait_stage = [VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT as VkPipelineStageFlags];
			let cmd_buffers = [self.cmdbuf];
			let num_acquire_semaphores;
			let num_release_semaphores;
			let acquire_semaphores_ptr: *const VkSemaphore;
			let release_semaphores_ptr: *const VkSemaphore;
			if let Some(swapchain_image_index) = self.swapchain_image_index {
				let swapchain_image = ctx.get_swapchain_image(swapchain_image_index);
				num_acquire_semaphores = 1;
				num_release_semaphores = 1;
				acquire_semaphores_ptr = &swapchain_image.acquire_semaphore.get_vk_semaphore();
				release_semaphores_ptr = &swapchain_image.release_semaphore.get_vk_semaphore();
			} else {
				num_acquire_semaphores = 0;
				num_release_semaphores = 0;
				acquire_semaphores_ptr = null();
				release_semaphores_ptr = null();
			}
			let submit_info = VkSubmitInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_SUBMIT_INFO,
				pNext: null(),
				waitSemaphoreCount: num_acquire_semaphores,
				pWaitSemaphores: acquire_semaphores_ptr,
				pWaitDstStageMask: wait_stage.as_ptr(),
				commandBufferCount: 1,
				pCommandBuffers: cmd_buffers.as_ptr(),
				signalSemaphoreCount: num_release_semaphores,
				pSignalSemaphores: release_semaphores_ptr,
			};
			let queue = if let Some(queue_index) = self.queue_index {
				ctx.get_vk_queue(queue_index)
			} else {
				let mut queue_index = 0;
				ctx.get_any_vk_queue_anyway(&mut queue_index)
			};
			vkcore.vkQueueSubmit(*queue, 1, &submit_info, self.submit_fence)?;
			self.submitted = true;
			Ok(())
		} else {
			panic!("Duplicated call to `VulkanCommandPoolInUse::submit()`, please set the `submitted` member to false to re-submit again if you wish.")
		}
	}

	/// End recording to the command buffer and submit the commands to the queue
	pub fn end(self) {}
}

impl Drop for VulkanCommandPoolInUse<'_> {
	fn drop(&mut self) {
		if !self.submitted {
			self.submit().unwrap();
		}
		let ctx = self.ctx.lock().unwrap();
		if !self.one_time_submit {
			let vkcore = ctx.get_vkcore();
			vkcore.vkResetCommandBuffer(self.cmdbuf, 0).unwrap();
		}
	}
}
