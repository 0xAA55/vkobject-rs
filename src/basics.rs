
#![allow(clippy::uninit_vec)]
#![allow(clippy::too_many_arguments)]
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::{MaybeUninit, transmute},
	ptr::{null, null_mut},
	sync::{Mutex, Arc, Weak},
};

#[derive(Debug, Clone)]
pub enum VulkanError {
	VkError(VkError),
	ChooseGpuFailed,
	NoGoodQueueForSurface(&'static str),
}

impl From<VkError> for VulkanError {
	fn from(e: VkError) -> Self {
		Self::VkError(e)
	}
}

#[derive(Debug, Clone)]
pub struct VulkanGpuInfo {
	gpu: VkPhysicalDevice,
	properties: VkPhysicalDeviceProperties,
	queue_families: Vec<VkQueueFamilyProperties>,
	extension_properties: Vec<VkExtensionProperties>,
}

impl VulkanGpuInfo {
	pub fn get_gpu_info(vkcore: &VkCore) -> Result<Vec<VulkanGpuInfo>, VkError> {
		let mut num_gpus = 0u32;
		vkcore.vkEnumeratePhysicalDevices(vkcore.instance, &mut num_gpus, null_mut())?;
		let mut gpus = Vec::<VkPhysicalDevice>::with_capacity(num_gpus as usize);
		vkcore.vkEnumeratePhysicalDevices(vkcore.instance, &mut num_gpus, gpus.as_mut_ptr())?;
		unsafe {gpus.set_len(num_gpus as usize)};
		let mut ret = Vec::<VulkanGpuInfo>::with_capacity(num_gpus as usize);
		for gpu in gpus {
			let mut properties: VkPhysicalDeviceProperties = unsafe {MaybeUninit::zeroed().assume_init()};
			vkcore.vkGetPhysicalDeviceProperties(gpu, &mut properties)?;
			let mut num_queue_families = 0u32;
			vkcore.vkGetPhysicalDeviceQueueFamilyProperties(gpu, &mut num_queue_families, null_mut())?;
			let mut queue_families = Vec::<VkQueueFamilyProperties>::with_capacity(num_queue_families as usize);
			vkcore.vkGetPhysicalDeviceQueueFamilyProperties(gpu, &mut num_queue_families, queue_families.as_mut_ptr())?;
			unsafe {queue_families.set_len(num_queue_families as usize)};
			let mut num_extension_properties = 0u32;
			vkcore.vkEnumerateDeviceExtensionProperties(gpu, null(), &mut num_extension_properties, null_mut())?;
			let mut extension_properties = Vec::<VkExtensionProperties>::with_capacity(num_extension_properties as usize);
			vkcore.vkEnumerateDeviceExtensionProperties(gpu, null(), &mut num_extension_properties, extension_properties.as_mut_ptr())?;
			unsafe {extension_properties.set_len(num_extension_properties as usize)};
			ret.push(VulkanGpuInfo {
				gpu,
				properties,
				queue_families,
				extension_properties,
			});
		}
		Ok(ret)
	}

	pub fn get_vk_physical_device(&self) -> VkPhysicalDevice {
		self.gpu
	}

	pub fn get_queue_families(&self) -> &[VkQueueFamilyProperties] {
		self.queue_families.as_ref()
	}

	pub fn get_queue_family_index(&self, queue_flag_match: u32) -> u32 {
		for i in 0..self.queue_families.len() {
			if (self.queue_families[i].queueFlags & queue_flag_match) == queue_flag_match {
				return i as u32;
			}
		}
		u32::MAX
	}

	pub fn get_properties(&self) -> &VkPhysicalDeviceProperties {
		&self.properties
	}

	pub fn get_extension_properties(&self) -> &[VkExtensionProperties] {
		self.extension_properties.as_ref()
	}
}

#[derive(Debug)]
pub struct VulkanSemaphore {
	ctx: Weak<Mutex<VulkanContext>>,
	semaphore: VkSemaphore,
}

unsafe impl Send for VulkanSemaphore {}

impl VulkanSemaphore {
	pub fn new(vkcore: &VkCore, device: &VulkanDevice) -> Result<Self, VkError> {
		let ci = VkSemaphoreCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			pNext: null(),
			flags: 0,
		};
		let mut semaphore: VkSemaphore = null();
		vkcore.vkCreateSemaphore(device.get_vk_device(), &ci, null(), &mut semaphore)?;
		Ok(Self{
			ctx: Weak::new(),
			semaphore,
		})
	}

	pub fn get_vk_semaphore(&self) -> VkSemaphore {
		self.semaphore
	}

	fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.ctx = ctx;
	}
}

impl Drop for VulkanSemaphore {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			vkcore.vkDestroySemaphore(ctx.get_vk_device(), self.semaphore, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanFence {
	ctx: Weak<Mutex<VulkanContext>>,
	fence: VkFence,
}

unsafe impl Send for VulkanFence {}

impl VulkanFence {
	pub fn new(vkcore: &VkCore, device: &VulkanDevice) -> Result<Self, VkError> {
		let ci = VkFenceCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			pNext: null(),
			flags: 0,
		};
		let mut fence: VkFence = null();
		vkcore.vkCreateFence(device.get_vk_device(), &ci, null(), &mut fence)?;
		Ok(Self{
			ctx: Weak::new(),
			fence,
		})
	}

	pub fn get_vk_fence(&self) -> VkFence {
		self.fence
	}

	fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.ctx = ctx;
	}
}

impl Drop for VulkanFence {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			vkcore.vkDestroyFence(ctx.get_vk_device(), self.fence, null()).unwrap();
		}
	}
}
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

#[derive(Debug, Clone)]
pub struct VulkanContext {
	vkcore: Arc<VkCore>,
	pub device: Arc<VulkanDevice>,
	pub surface: Arc<Mutex<VulkanSurface>>,
	pub swapchain: Arc<Mutex<VulkanSwapchain>>,
	pub cmdpools: Vec<Arc<Mutex<VulkanCommandPool>>>,
	cur_swapchain_image_index: u32,
}

unsafe impl Send for VulkanContext {}

impl VulkanContext {
	/// Create a new `VulkanContext`
	pub fn new(vkcore: Arc<VkCore>, device: Arc<VulkanDevice>, surface: Arc<Mutex<VulkanSurface>>, width: u32, height: u32, vsync: bool, max_concurrent_frames: usize, is_vr: bool) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let mut cmdpools = Vec::<Arc<Mutex<VulkanCommandPool>>>::with_capacity(max_concurrent_frames);
		for _ in 0..max_concurrent_frames {
			cmdpools.push(Arc::new(Mutex::new(VulkanCommandPool::new(&vkcore, &device)?)));
		}
		let ret = Arc::new(Mutex::new(Self{
			vkcore: vkcore.clone(),
			device: device.clone(),
			surface: surface.clone(),
			swapchain: Arc::new(Mutex::new(VulkanSwapchain::new(&vkcore, &device, surface.clone(), width, height, vsync, is_vr)?)),
			cmdpools,
			cur_swapchain_image_index: 0,
		}));
		let weak = Arc::downgrade(&ret);
		if true {
			let mut lock = ret.lock().unwrap();
			lock.surface.lock().unwrap().ctx = weak.clone();
			lock.swapchain.lock().unwrap().set_ctx(weak.clone());
			for cmdpool in lock.cmdpools.iter_mut() {
				cmdpool.lock().unwrap().ctx = weak.clone();
			}
		}
		Ok(ret)
	}

	/// Get the Vulkan instance
	pub fn get_instance(&self) -> VkInstance {
		self.vkcore.instance
	}

	/// get the `VkCore`
	fn get_vkcore(&self) -> &VkCore {
		&self.vkcore
	}

	/// Get the current physical device
	pub fn get_vk_physical_device(&self) -> VkPhysicalDevice {
		self.device.get_vk_physical_device()
	}

	/// Get the current device
	pub fn get_vk_device(&self) -> VkDevice {
		self.device.get_vk_device()
	}

	/// Get the current queue for the current device
	pub fn get_vk_queue(&self) -> VkQueue {
		self.device.get_vk_queue()
	}

	/// Get the current surface
	pub fn get_vk_surface(&self) -> VkSurfaceKHR {
		let surface = self.surface.lock().unwrap();
		surface.get_vk_surface()
	}

	/// Get the current surface format
	pub fn get_vk_surface_format(&self) -> VkSurfaceFormatKHR {
		let surface = self.surface.lock().unwrap();
		*surface.get_vk_surface_format()
	}

	/// Get the current swapchain image index
	pub fn get_swapchain_image_index(&self) -> u32 {
		self.cur_swapchain_image_index
	}
}

