
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

