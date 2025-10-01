
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	ptr::null,
	sync::{
		Arc,
		atomic::{
			AtomicUsize,
			Ordering
		},
		Mutex,
		MutexGuard
	},
};

/// The Vulkan command pool, and the associated buffers, fence. Support multiple buffers; you can use one buffer for command recording and another for submitting to a queue, interleaved.
pub struct VulkanCommandPool {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the command pool
	pub(crate) pool: Mutex<VkCommandPool>,

	/// The command buffers of the command pool
	pub(crate) cmd_buffers: Vec<VkCommandBuffer>,

	/// The index of the command buffer
	pub index_of_buffer: AtomicUsize,

	/// The fence for the command pool
	pub submit_fence: Arc<VulkanFence>,
}

unsafe impl Send for VulkanCommandPool {}
unsafe impl Sync for VulkanCommandPool {}

impl VulkanCommandPool {
	/// Create a new `VulkanCommandPool`
	pub fn new(device: Arc<VulkanDevice>, num_buffers: usize) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vk_device = device.get_vk_device();
		let pool_ci = VkCommandPoolCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			pNext: null(),
			queueFamilyIndex: device.get_queue_family_index(),
			flags:
				VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT as VkCommandPoolCreateFlags |
				VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_TRANSIENT_BIT as VkCommandPoolCreateFlags,
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
		let pool = Mutex::new(pool);
		let mut cmd_buffers: Vec<VkCommandBuffer> = Vec::with_capacity(num_buffers);
		vkcore.vkAllocateCommandBuffers(vk_device, &cmd_buffers_ci, cmd_buffers.as_mut_ptr())?;
		unsafe {cmd_buffers.set_len(num_buffers)};
		let submit_fence = Arc::new(VulkanFence::new(device.clone())?);
		Ok(Self{
			device,
			pool,
			index_of_buffer: AtomicUsize::new(0),
			cmd_buffers,
			submit_fence,
		})
	}

	/// Use a command buffer of the command pool to record draw commands
	pub(crate) fn use_pool<'a>(&'a mut self, rt_props: Option<Arc<RenderTargetProps>>) -> Result<VulkanCommandPoolInUse<'a>, VulkanError> {
		let pool_lock = self.pool.lock().unwrap();
		let begin_info = VkCommandBufferBeginInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			pNext: null(),
			flags: 0,
			pInheritanceInfo: null(),
		};
		let buffer_index = self.index_of_buffer.fetch_add(1, Ordering::Relaxed) % self.cmd_buffers.len();
		self.device.vkcore.vkResetCommandBuffer(self.cmd_buffers[buffer_index], 0)?;
		self.device.vkcore.vkBeginCommandBuffer(self.cmd_buffers[buffer_index], &begin_info)?;
		VulkanCommandPoolInUse::new(self, pool_lock, self.cmd_buffers[buffer_index], rt_props)
	}

	/// Wait for the submit fence to be signaled
	pub fn wait_for_submit(&self, timeout: u64) -> Result<(), VulkanError> {
		self.submit_fence.wait(timeout)?;
		self.submit_fence.unsignal()?;
		Ok(())
	}
}

impl Debug for VulkanCommandPool {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanCommandPool")
		.field("pool", &self.pool)
		.field("cmd_buffers", &self.cmd_buffers)
		.field("index_of_buffer", &self.index_of_buffer)
		.field("submit_fence", &self.submit_fence)
		.finish()
	}
}

impl Drop for VulkanCommandPool {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		proceed_run(self.submit_fence.wait(u64::MAX));
		proceed_run(self.submit_fence.unsignal());
		proceed_run(vkcore.vkDestroyCommandPool(self.device.get_vk_device(), *self.pool.lock().unwrap(), null()));
	}
}

/// The RAII wrapper for the usage of a Vulkan command pool/buffer. When created, your command could be recorded to the command buffer.
pub struct VulkanCommandPoolInUse<'a> {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The command buffer we are using here
	pub(crate) cmdbuf: VkCommandBuffer,

	/// The command pool to submit commands
	pub(crate) pool_lock: MutexGuard<'a, VkCommandPool>,

	/// The render target for the command pool to draw to
	pub rt_props: Option<Arc<RenderTargetProps>>,

	/// The fence indicating if all commands were submitted
	pub submit_fence: Arc<VulkanFence>,

	/// Is recording commands ended
	pub(crate) ended: bool,

	/// Is the commands submitted
	pub submitted: bool,
}

impl<'a> VulkanCommandPoolInUse<'a> {
	/// Create a RAII binding to the `VulkanCommandPool` in use
	fn new(cmdpool: &VulkanCommandPool, pool_lock: MutexGuard<'a, VkCommandPool>, cmdbuf: VkCommandBuffer, rt_props: Option<Arc<RenderTargetProps>>) -> Result<Self, VulkanError> {
		let device = cmdpool.device.clone();
		let submit_fence = cmdpool.submit_fence.clone();
		Ok(Self {
			device,
			cmdbuf,
			pool_lock,
			rt_props,
			submit_fence,
			ended: false,
			submitted: false,
		})
	}

	/// Get the current command buffer
	pub(crate) fn get_vk_cmdbuf(&self) -> VkCommandBuffer {
		self.cmdbuf
	}

	/// End recording commands
	pub fn end_cmd(&mut self) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		if !self.ended {
			vkcore.vkEndCommandBuffer(self.cmdbuf)?;
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
			let wait_stage = [VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT as VkPipelineStageFlags];
			let cmd_buffers = [self.cmdbuf];

			let (acquire_semaphores, release_semaphores) = if let Some(rt_props) = &self.rt_props {
				(
					vec![rt_props.acquire_semaphore.lock().unwrap().get_vk_semaphore()],
					vec![rt_props.release_semaphore.get_vk_semaphore()]
				)
			} else {
				(vec![], vec![])
			};
			let submit_info = VkSubmitInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_SUBMIT_INFO,
				pNext: null(),
				waitSemaphoreCount: acquire_semaphores.len() as u32,
				pWaitSemaphores: acquire_semaphores.as_ptr(),
				pWaitDstStageMask: wait_stage.as_ptr(),
				commandBufferCount: 1,
				pCommandBuffers: cmd_buffers.as_ptr(),
				signalSemaphoreCount: release_semaphores.len() as u32,
				pSignalSemaphores: release_semaphores.as_ptr(),
			};
			let submits = [submit_info];
			self.submit_fence.wait(u64::MAX)?;
			self.submit_fence.unsignal()?;
			vkcore.vkQueueSubmit(self.device.get_vk_queue(), submits.len() as u32, submits.as_ptr(), self.submit_fence.get_vk_fence())?;
			self.submit_fence.set_is_being_signaled();
			self.submitted = true;
			Ok(())
		} else {
			panic!("Duplicated call to `VulkanCommandPoolInUse::submit()`, please set the `submitted` member to false to re-submit again if you wish.")
		}
	}

	/// End recording to the command buffer and submit the commands to the queue
	pub fn end(self) {}
}

impl Debug for VulkanCommandPoolInUse<'_> {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanCommandPoolInUse")
		.field("cmdbuf", &self.cmdbuf)
		.field("pool_lock", &self.pool_lock)
		.field("rt_props", &self.rt_props)
		.field("submit_fence", &self.submit_fence)
		.field("ended", &self.ended)
		.field("submitted", &self.submitted)
		.finish()
	}
}

impl Drop for VulkanCommandPoolInUse<'_> {
	fn drop(&mut self) {
		if !self.submitted {
			proceed_run(self.submit())
		}
	}
}
