
use crate::prelude::*;
use std::{
	fmt::Debug,
	mem::MaybeUninit,
	sync::{Mutex, Arc, MutexGuard},
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

	pub vsync: bool,
	pub max_concurrent_frames: usize,
	pub is_vr: bool,
}

#[derive(Debug)]
pub struct VulkanContext {
	pub(crate) vkcore: Arc<VkCore>,
	pub(crate) device: Arc<VulkanDevice>,
	pub(crate) surface: Arc<Mutex<VulkanSurface>>,
	pub(crate) swapchain: VulkanSwapchain,
	pub(crate) cmdpools: Vec<VulkanCommandPool>,
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
		let size = Self::get_surface_size_(vkcore, device, surface.clone())?;
		let swapchain = VulkanSwapchain::new(vkcore, device, surface.clone(), size.width, size.height, create_info.vsync, create_info.is_vr, None)?;
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
	pub(crate) fn get_instance(&self) -> VkInstance {
		self.vkcore.get_instance()
	}

	/// get the `VkCore`
	pub(crate) fn get_vkcore(&self) -> &VkCore {
		&self.vkcore
	}

	/// Get the current physical device
	pub(crate) fn get_vk_physical_device(&self) -> VkPhysicalDevice {
		self.device.get_vk_physical_device()
	}

	/// Get the current device
	pub(crate) fn get_vk_device(&self) -> VkDevice {
		self.device.get_vk_device()
	}

	/// Get a queue for the current device
	pub(crate) fn get_vk_queue(&self, queue_index: usize) -> MutexGuard<'_, VkQueue> {
		self.device.get_vk_queue(queue_index)
	}

	/// Get the current surface
	pub(crate) fn get_vk_surface(&self) -> VkSurfaceKHR {
		self.surface.lock().unwrap().get_vk_surface()
	}

	/// Get the current surface format
	pub(crate) fn get_vk_surface_format(&self) -> VkSurfaceFormatKHR {
		*self.surface.lock().unwrap().get_vk_surface_format()
	}

	pub(crate) fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.swapchain.get_vk_swapchain()
	}

	pub(crate) fn get_swapchain_extent(&self) -> VkExtent2D {
		self.swapchain.get_swapchain_extent()
	}

	pub(crate) fn get_swapchain_image(&self, index: usize) -> &VulkanSwapchainImage {
		self.swapchain.get_image(index)
	}

	pub fn get_surface_size_(vkcore: &VkCore, device: &VulkanDevice, surface: Arc<Mutex<VulkanSurface>>) -> Result<VkExtent2D, VulkanError> {
		let mut surface_properties: VkSurfaceCapabilitiesKHR = unsafe {MaybeUninit::zeroed().assume_init()};
		let surface = surface.lock().unwrap();
		vkcore.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.get_vk_physical_device(), surface.get_vk_surface(), &mut surface_properties)?;
		Ok(surface_properties.currentExtent)
	}

	pub fn get_surface_size(&self) -> Result<VkExtent2D, VulkanError> {
		Self::get_surface_size_(&self.vkcore, &self.device, self.surface.clone())
	}

	pub fn on_resize(&mut self) -> Result<bool, VulkanError> {
		let surface_size = self.get_surface_size()?;
		let swapchain_extent = self.get_swapchain_extent();
		if	swapchain_extent.width == surface_size.width &&
			swapchain_extent.height == surface_size.height {
			Ok(false)
		} else {
			self.device.wait_idle()?;
			let prev_chain = self.get_vk_swapchain();
			let new_chain = VulkanSwapchain::new(&self.vkcore, &self.device, self.surface.clone(), surface_size.width, surface_size.height, self.swapchain.get_is_vsync(), self.swapchain.get_is_vr(), Some(prev_chain))?;
			self.swapchain = new_chain;
			Ok(true)
		}
	}

	/// Get the current swapchain image index
	pub fn get_swapchain_image_index(&self) -> u32 {
		self.swapchain.get_image_index()
	}
}

