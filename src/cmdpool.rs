
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::{MaybeUninit, transmute},
	ptr::{null, null_mut},
	sync::{Mutex, Arc, Weak},
};

#[derive(Debug)]
pub struct VulkanCommandPool {
	ctx: Weak<Mutex<VulkanContext>>,
	pool: VkCommandPool,
	cmd_buffer: VkCommandBuffer,
	fence: VkFence,
}

unsafe impl Send for VulkanCommandPool {}

impl VulkanCommandPool {
	pub fn new(vkcore: &VkCore, device: &VulkanDevice) -> Result<Self, VkError> {
		let vk_device = device.get_vk_device();
		let pool_ci = VkCommandPoolCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			pNext: null(),
			queueFamilyIndex: device.queue_family_index,
			flags: VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT as u32,
		};
		let mut pool: VkCommandPool = null();
		vkcore.vkCreateCommandPool(vk_device, &pool_ci, null(), &mut pool)?;
		let cmd_buffers_ci = VkCommandBufferAllocateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			pNext: null(),
			commandPool: pool,
			level: VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			commandBufferCount: 1,
		};
		let mut cmd_buffer: VkCommandBuffer = null();
		vkcore.vkAllocateCommandBuffers(vk_device, &cmd_buffers_ci, &mut cmd_buffer)?;
		let mut fence: VkFence = null();
		let fence_ci = VkFenceCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			pNext: null(),
			flags: VkFenceCreateFlagBits::VK_FENCE_CREATE_SIGNALED_BIT as u32,
		};
		vkcore.vkCreateFence(vk_device, &fence_ci, null(), &mut fence)?;
		Ok(Self{
			ctx: Weak::new(),
			pool,
			cmd_buffer,
			fence,
		})
	}

	/// Retrieve the command pool
	pub fn get_vk_cmdpool(&self) -> VkCommandPool {
		self.pool
	}

	/// Get the command buffers
	pub fn get_vk_cmd_buffer(&self) -> VkCommandBuffer {
		self.cmd_buffer
	}

	/// Get the fences
	pub fn get_vk_fence(&self) -> VkFence {
		self.fence
	}
}

impl Drop for VulkanCommandPool {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			let device = ctx.get_vk_device();
			vkcore.vkDestroyFence(device, self.fence, null()).unwrap();
			vkcore.vkDestroyCommandPool(ctx.get_vk_device(), self.pool, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanCommandPoolInUse<'a, 'b> {
	ctx: Arc<Mutex<VulkanContext>>,
	cmdpool: &'a VulkanCommandPool,
	swapchain_image: &'b VulkanSwapchainImage,
	one_time_submit: bool,
	ended: bool,
	pub submitted: bool,
}

impl<'a, 'b> VulkanCommandPoolInUse<'a, 'b> {
	pub fn new(cmdpool: &'a VulkanCommandPool, swapchain_image: &'b VulkanSwapchainImage, one_time_submit: bool) -> Result<Self, VkError> {
		let ctx = cmdpool.ctx.upgrade().unwrap();
		let ctx_g = ctx.lock().unwrap();
		let vkcore = ctx_g.get_vkcore();
		let cmdbuf = cmdpool.get_vk_cmd_buffer();
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
			swapchain_image,
			one_time_submit,
			ended: false,
			submitted: false,
		})
	}

	pub fn end_cmd(&mut self) -> Result<(), VkError> {
		if !self.ended {
			let ctx = self.ctx.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			let cmdbuf = self.cmdpool.get_vk_cmd_buffer();
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

	pub fn submit(&mut self) -> Result<(), VkError> {
		if !self.ended {
			self.end_cmd()?;
		}
		if !self.submitted {
			let ctx = self.ctx.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			let cmdbuf = self.cmdpool.get_vk_cmd_buffer();

			let wait_stage = [VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT as VkPipelineStageFlags];
			let cmd_buffers = [cmdbuf];
			let submit_info = VkSubmitInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_SUBMIT_INFO,
				pNext: null(),
				waitSemaphoreCount: 1,
				pWaitSemaphores: &self.swapchain_image.acquire_semaphore.get_vk_semaphore(),
				pWaitDstStageMask: wait_stage.as_ptr(),
				commandBufferCount: 1,
				pCommandBuffers: cmd_buffers.as_ptr(),
				signalSemaphoreCount: 1,
				pSignalSemaphores: &self.swapchain_image.release_semaphore.get_vk_semaphore(),
			};
			vkcore.vkQueueSubmit(ctx.get_vk_queue(), 1, &submit_info, self.swapchain_image.queue_submit_fence.get_vk_fence())?;
			self.submitted = true;
			Ok(())
		} else {
			panic!("Duplicated call to `VulkanCommandPoolInUse::submit()`, please set the `submitted` member to false to re-submit again if you wish.")
		}
	}

	pub fn end(self) {}
}

impl Drop for VulkanCommandPoolInUse<'_, '_> {
	fn drop(&mut self) {
		if !self.submitted {
			self.submit().unwrap();
		}
		if !self.one_time_submit {
			let ctx = self.ctx.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			let cmdbuf = self.cmdpool.get_vk_cmd_buffer();
			vkcore.vkResetCommandBuffer(cmdbuf, 0).unwrap();
		}
	}
}