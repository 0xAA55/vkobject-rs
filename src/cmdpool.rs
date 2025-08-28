
use crate::prelude::*;
use std::{
	fmt::Debug,
	ptr::null,
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
		Mutex,
		Weak
	},
};

/// The Vulkan command pool, and the associated buffers, fence. Support multiple buffers; you can use one buffer for command recording and another for submitting to a queue, interleaved.
#[derive(Debug)]
pub struct VulkanCommandPool {
	/// The `VulkanContext` that helps to manage the resources of the command pool
	ctx: Weak<Mutex<VulkanContext>>,

	/// The handle to the command pool
	pool: VkCommandPool,

	/// The command buffers of the command pool
	cmd_buffers: Vec<VkCommandBuffer>,

	/// The last command buffer index
	pub last_buf_index: u32,

	/// The fence for the command pool
	pub(crate) fence: VulkanFence,

	/// Is the command pool in use now?
	pub(crate) is_inuse: AtomicBool,
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
			pool,
			cmd_buffers,
			last_buf_index: 0,
			fence: VulkanFence::new_(vkcore, vk_device)?,
			is_inuse: AtomicBool::new(false),
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
	pub(crate) fn get_vk_cmdpool(&self) -> VkCommandPool {
		self.pool
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

	/// Use a command buffer
	pub fn use_buf<'a>(&'a mut self, queue_index: usize, swapchain_image_index: usize, one_time_submit: bool) -> Result<VulkanCommandPoolInUse<'a>, VulkanError> {
		if let Err(_) = self.is_inuse.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed) {
			return Err(VulkanError::CommandPoolIsInUse);
		}
		let cmdbuf_index = self.last_buf_index as usize;
		self.last_buf_index += 1;
		if self.last_buf_index as usize > self.cmd_buffers.len() {
			self.last_buf_index = 0;
		}
		VulkanCommandPoolInUse::new(self, cmdbuf_index, queue_index, swapchain_image_index, one_time_submit)
	}
}

impl Drop for VulkanCommandPool {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			vkcore.vkDestroyCommandPool(ctx.get_vk_device(), self.pool, null()).unwrap();
		}
	}
}

/// The RAII wrapper for the usage of a Vulkan command pool/buffer. When created, your command could be recorded to the command buffer.
#[derive(Debug)]
pub struct VulkanCommandPoolInUse<'a> {
	/// The `VulkanContext` that helps to manage the resources of the command pool
	pub(crate) ctx: Arc<Mutex<VulkanContext>>,

	/// The command pool we are using here
	pub(crate) cmdpool: &'a VulkanCommandPool,

	/// The command buffer index using right now
	cmdbuf_index: usize,

	/// The queue index for the command pool to submit
	queue_index: usize,

	/// The swapchain image index for the command pool to draw to
	swapchain_image_index: usize,

	/// Is this command buffer got automatically cleaned when submitted
	pub(crate) one_time_submit: bool,

	/// Is recording commands ended
	pub(crate) ended: bool,

	/// Is the commands submitted
	pub(crate) submitted: bool,
}

impl<'a> VulkanCommandPoolInUse<'a> {
	/// Create a RAII binding to the `VulkanCommandPool` in use
	fn new(cmdpool: &'a VulkanCommandPool, cmdbuf_index: usize, queue_index: usize, swapchain_image_index: usize, one_time_submit: bool) -> Result<Self, VulkanError> {
		let ctx = cmdpool.ctx.upgrade().unwrap();
		let ctx_g = ctx.lock().unwrap();
		let vkcore = ctx_g.get_vkcore();
		let cmdbuf = cmdpool.get_vk_cmd_buffers()[cmdbuf_index];
		let begin_info = VkCommandBufferBeginInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			pNext: null(),
			flags: if one_time_submit {VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT as u32} else {0u32},
			pInheritanceInfo: null(),
		};
		vkcore.vkBeginCommandBuffer(cmdbuf, &begin_info)?;
		Ok(Self {
			ctx: ctx.clone(),
			cmdpool,
			cmdbuf_index,
			queue_index,
			swapchain_image_index,
			one_time_submit,
			ended: false,
			submitted: false,
		})
	}

	/// Get the current command buffer
	pub(crate) fn get_vk_cmdbuf(&self) -> VkCommandBuffer {
		self.cmdpool.get_vk_cmd_buffers()[self.cmdbuf_index]
	}

	/// Get the current command buffer index
	pub fn get_cmdbuf_index(&self) -> usize {
		self.cmdbuf_index
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

			let swapchain_image = ctx.get_swapchain_image(self.swapchain_image_index);
			let wait_stage = [VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT as VkPipelineStageFlags];
			let cmd_buffers = [self.get_vk_cmdbuf()];
			let submit_info = VkSubmitInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_SUBMIT_INFO,
				pNext: null(),
				waitSemaphoreCount: 1,
				pWaitSemaphores: &swapchain_image.acquire_semaphore.get_vk_semaphore(),
				pWaitDstStageMask: wait_stage.as_ptr(),
				commandBufferCount: 1,
				pCommandBuffers: cmd_buffers.as_ptr(),
				signalSemaphoreCount: 1,
				pSignalSemaphores: &swapchain_image.release_semaphore.get_vk_semaphore(),
			};
			vkcore.vkQueueSubmit(*ctx.get_vk_queue(self.queue_index), 1, &submit_info, swapchain_image.queue_submit_fence.get_vk_fence())?;
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
		if !self.one_time_submit {
			let ctx = self.ctx.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			let cmdbuf = self.cmdpool.get_vk_cmd_buffers()[self.get_cmdbuf_index()];
			vkcore.vkResetCommandBuffer(cmdbuf, 0).unwrap();
		}
		self.cmdpool.is_inuse.store(false, Ordering::Release);
	}
}
