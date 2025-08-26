
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::{MaybeUninit, transmute},
	ptr::{null, null_mut},
	sync::{Mutex, Arc, Weak},
};

#[derive(Debug)]
pub struct VulkanContextCreateInfo<'a> {
	pub vkcore: Arc<VkCore>,
	pub device: VulkanDevice,

	#[cfg(any(feature = "glfw", test))]
	pub window: &'a glfw::PWindow,

	#[cfg(feature = "win32_khr")]
	pub hwnd: HWND,
	#[cfg(feature = "win32_khr")]
	pub hinstance: HINSTANCE,

	#[cfg(feature = "android_khr")]
	pub window: *const ANativeWindow,

	#[cfg(feature = "ios_mvk")]
	pub view: *const c_void,

	#[cfg(feature = "macos_mvk")]
	pub view: *const c_void,

	#[cfg(feature = "metal_ext")]
	pub metal_layer: *const CAMetalLayer,

	#[cfg(feature = "wayland_khr")]
	pub display: *const c_void,
	#[cfg(feature = "wayland_khr")]
	pub surface: *const c_void,

	#[cfg(feature = "xcb_khr")]
	pub connection: *const c_void,
	#[cfg(feature = "xcb_khr")]
	pub window: xcb_window_t,

	pub width: u32,
	pub height: u32,
	pub vsync: bool,
	pub max_concurrent_frames: usize,
	pub is_vr: bool,
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
	pub fn new(create_info: VulkanContextCreateInfo) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let vkcore = &create_info.vkcore;
		let device = &create_info.device;

		#[cfg(any(feature = "glfw", test))]
		let surface = VulkanSurface::new(vkcore, device, create_info.window)?;
		#[cfg(feature = "win32_khr")]
		let surface = VulkanSurface::new(vkcore, device, create_info.hwnd, create_info.hinstance)?;
		#[cfg(feature = "android_khr")]
		let surface = VulkanSurface::new(vkcore, device, create_info.window)?;
		#[cfg(feature = "ios_mvk")]
		let surface = VulkanSurface::new(vkcore, device, create_info.view)?;
		#[cfg(feature = "macos_mvk")]
		let surface = VulkanSurface::new(vkcore, device, create_info.view)?;
		#[cfg(feature = "metal_ext")]
		let surface = VulkanSurface::new(vkcore, device, create_info.metal_layer)?;
		#[cfg(feature = "wayland_khr")]
		let surface = VulkanSurface::new(vkcore, device, create_info.display, create_info.surface)?;
		#[cfg(feature = "xcb_khr")]
		let surface = VulkanSurface::new(vkcore, device, create_info.connection, create_info.window)?;

		let mut cmdpools = Vec::<VulkanCommandPool>::with_capacity(create_info.max_concurrent_frames);
		for _ in 0..create_info.max_concurrent_frames {
			cmdpools.push(VulkanCommandPool::new(vkcore, device)?);
		}
		let swapchain = VulkanSwapchain::new(vkcore, device, surface.clone(), create_info.width, create_info.height, create_info.vsync, create_info.is_vr)?;
		let ret = Arc::new(Mutex::new(Self {
			vkcore: create_info.vkcore,
			device: Arc::new(create_info.device),
			surface,
			swapchain,
			cmdpools,
			cur_swapchain_image_index: 0,
		}));
		let weak = Arc::downgrade(&ret);
		let mut lock = ret.lock().unwrap();
		lock.surface.lock().unwrap().set_ctx(weak.clone());
		lock.swapchain.set_ctx(weak.clone());
		for cmdpool in lock.cmdpools.iter_mut() {
			cmdpool.set_ctx(weak.clone());
		}
		drop(lock);
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

	pub(crate) fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.swapchain.get_vk_swapchain()
	}

	/// Get the current swapchain image index
	pub fn get_swapchain_image_index(&self) -> u32 {
		self.cur_swapchain_image_index
	}

	/// Get another swapchain image index
	pub fn switch_next_swapchain_image_index(&mut self) -> u32 {
		self.cur_swapchain_image_index += 1;
		if self.cur_swapchain_image_index as usize >= self.swapchain.images.len() {
			self.cur_swapchain_image_index = 0;
		}
		self.cur_swapchain_image_index
	}
}

