
use crate::prelude::*;
use std::{
	fmt::Debug,
	ptr::null,
	sync::{Mutex, Arc, Weak},
};

#[derive(Debug)]
pub struct VulkanCommandPool {
	ctx: Weak<Mutex<VulkanContext>>,
	pool: VkCommandPool,
	cmd_buffers: Vec<VkCommandBuffer>,
	pub last_buf_index: Mutex<u32>,
	pub(crate) fence: VulkanFence,
}

unsafe impl Send for VulkanCommandPool {}

impl VulkanCommandPool {
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, num_buffers: usize) -> Result<Self, VulkanError> {
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
			last_buf_index: Mutex::new(0),
			fence: VulkanFence::new_(vkcore, vk_device)?,
		})
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
	pub fn use_buf<'a>(&'a mut self, swapchain_image_index: usize, one_time_submit: bool) -> Result<VulkanCommandPoolInUse<'a>, VulkanError> {
		let mut lock = self.last_buf_index.lock().unwrap();
		let index = *lock as usize;
		*lock += 1;
		if *lock as usize > self.cmd_buffers.len() {
			*lock = 0;
		}
		drop(lock);
		VulkanCommandPoolInUse::new(self, index, swapchain_image_index, one_time_submit)
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

#[derive(Debug)]
pub struct VulkanCommandPoolInUse<'a> {
	pub(crate) ctx: Arc<Mutex<VulkanContext>>,
	cmdpool: &'a VulkanCommandPool,
	cmdbuf_index: u32,
	swapchain_image_index: usize,
	pub(crate) one_time_submit: bool,
	pub(crate) ended: bool,
	pub(crate) submitted: bool,
}

impl<'a, 'b> VulkanCommandPoolInUse<'a> {
	fn new(cmdpool: &'a VulkanCommandPool, cmdbuf_index: usize, swapchain_image_index: usize, one_time_submit: bool) -> Result<Self, VulkanError> {
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
			cmdbuf_index: cmdbuf_index as u32,
			swapchain_image_index,
			one_time_submit,
			ended: false,
			submitted: false,
		})
	}

	pub fn get_cmdbuf_index(&self) -> usize {
		self.cmdbuf_index as usize
	}

	pub fn is_one_time_submit(&self) -> bool {
		self.one_time_submit
	}

	pub fn end_cmd(&mut self) -> Result<(), VulkanError> {
		if !self.ended {
			let ctx = self.ctx.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			let cmdbuf = self.cmdpool.get_vk_cmd_buffers()[self.get_cmdbuf_index()];
			vkcore.vkEndCommandBuffer(cmdbuf)?;
			self.ended = true;
			Ok(())
		} else {
			panic!("Duplicated call to `VulkanCommandPoolInUse::end()`")
		}
	}

	pub fn is_ended(&self) -> bool {
		self.ended
	}

	pub fn submit(&mut self) -> Result<(), VulkanError> {
		if !self.ended {
			self.end_cmd()?;
		}
		if !self.submitted {
			let ctx = self.ctx.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			let cmdbuf = self.cmdpool.get_vk_cmd_buffers()[self.get_cmdbuf_index()];

			let swapchain_image = ctx.get_swapchain_image(self.swapchain_image_index);
			let wait_stage = [VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT as VkPipelineStageFlags];
			let cmd_buffers = [cmdbuf];
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
			vkcore.vkQueueSubmit(*ctx.get_vk_queue(self.swapchain_image_index), 1, &submit_info, swapchain_image.queue_submit_fence.get_vk_fence())?;
			self.submitted = true;
			Ok(())
		} else {
			panic!("Duplicated call to `VulkanCommandPoolInUse::submit()`, please set the `submitted` member to false to re-submit again if you wish.")
		}
	}

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
	}
}