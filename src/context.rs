
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::{MaybeUninit, transmute},
	ptr::{null, null_mut},
	sync::{Mutex, Arc, Weak},
};

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

