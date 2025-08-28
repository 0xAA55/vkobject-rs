
use crate::prelude::*;
use std::{
	fmt::Debug,
	mem::MaybeUninit,
	sync::{Mutex, Arc, MutexGuard},
};

#[derive(Debug)]
	#[cfg(any(feature = "glfw", test))]
	/// The window from GLFW
	pub window: &'a glfw::PWindow,

	#[cfg(feature = "win32_khr")]
	/// The Windows window handle
	pub hwnd: HWND,
	#[cfg(feature = "win32_khr")]
	/// The Windows application instance handle
	pub hinstance: HINSTANCE,

	#[cfg(feature = "android_khr")]
	/// The Android window
	pub window: *const ANativeWindow,

	#[cfg(feature = "ios_mvk")]
	/// The IOS view
	pub view: *const c_void,

	#[cfg(feature = "macos_mvk")]
	/// The MacOS view
	pub view: *const c_void,

	#[cfg(feature = "metal_ext")]
	/// The Metal layer
	pub metal_layer: *const CAMetalLayer,

	#[cfg(feature = "wayland_khr")]
	/// The Wayland display
	pub display: *const c_void,
	#[cfg(feature = "wayland_khr")]
	/// The Wayland surface
	pub surface: *const c_void,

	#[cfg(feature = "xcb_khr")]
	/// The XCB connection
	pub connection: *const c_void,
	#[cfg(feature = "xcb_khr")]
	/// The XCB window
	pub window: xcb_window_t,
/// The struct to create the `VulkanContext`
#[derive(Debug)]
pub struct VulkanContextCreateInfo<'a> {
	/// The most important thing: the Vulkan driver is provided here
	pub vkcore: Arc<VkCore>,

	/// The device to use
	pub device: VulkanDevice,


	/// Is VSYNC should be on
	pub vsync: bool,

	/// How many frames could be rendered concurrently
	pub max_concurrent_frames: usize,

	/// Is this a VR project?
	pub is_vr: bool,
}

#[derive(Debug)]
pub struct VulkanContext {
	/// The Vulkan driver
	pub(crate) vkcore: Arc<VkCore>,

	/// The device in use
	pub(crate) device: Arc<VulkanDevice>,

	/// The surface in use
	pub(crate) surface: Arc<Mutex<VulkanSurface>>,

	/// The swapchain
	pub(crate) swapchain: VulkanSwapchain,

	/// The command pools
	pub(crate) cmdpools: Vec<VulkanCommandPool>,
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
			cmdpools.push(VulkanCommandPool::new_(vkcore, device, 2)?);
		}
		let size = Self::get_surface_size_(vkcore, device, surface.clone())?;
		let swapchain = VulkanSwapchain::new(vkcore, device, surface.clone(), size.width, size.height, create_info.vsync, create_info.is_vr, None)?;
		let ret = Arc::new(Mutex::new(Self {
			vkcore: create_info.vkcore,
			device: Arc::new(create_info.device),
			surface,
			swapchain,
			cmdpools,
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

	/// Get the `VkPhysicalDevice` in use
	pub(crate) fn get_vk_physical_device(&self) -> VkPhysicalDevice {
		self.device.get_vk_physical_device()
	}

	/// Get the `VkDevice` in use
	pub(crate) fn get_vk_device(&self) -> VkDevice {
		self.device.get_vk_device()
	}

	/// Get a queue for the current device. To submit commands to a queue concurrently, the queue must be locked.
	pub(crate) fn get_vk_queue(&self, queue_index: usize) -> MutexGuard<'_, VkQueue> {
		self.device.get_vk_queue(queue_index)
	}

	/// Get the `VkSurfaceKHR`
	pub(crate) fn get_vk_surface(&self) -> VkSurfaceKHR {
		self.surface.lock().unwrap().get_vk_surface()
	}

	/// Get the current surface format
	pub(crate) fn get_vk_surface_format(&self) -> VkSurfaceFormatKHR {
		*self.surface.lock().unwrap().get_vk_surface_format()
	}

	/// Get the `VkSwapchainKHR`
	pub(crate) fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.swapchain.get_vk_swapchain()
	}

	/// Get the current swapchain extent(the framebuffer size)
	pub(crate) fn get_swapchain_extent(&self) -> VkExtent2D {
		self.swapchain.get_swapchain_extent()
	}

	/// Get the swapchain image by an index
	pub(crate) fn get_swapchain_image(&self, index: usize) -> &VulkanSwapchainImage {
		self.swapchain.get_image(index)
	}

	/// Get the current swapchain image index
	pub fn get_swapchain_image_index(&self) -> usize {
		self.swapchain.get_image_index() as usize
	}

	/// Get the swapchain image by the current index
	pub(crate) fn get_cur_swapchain_image(&self) -> &VulkanSwapchainImage {
		self.swapchain.get_image(self.get_swapchain_image_index())
	}

	/// Get the surface size, a.k.a. the frame buffer size
	pub fn get_surface_size_(vkcore: &VkCore, device: &VulkanDevice, surface: Arc<Mutex<VulkanSurface>>) -> Result<VkExtent2D, VulkanError> {
		let mut surface_properties: VkSurfaceCapabilitiesKHR = unsafe {MaybeUninit::zeroed().assume_init()};
		let surface = surface.lock().unwrap();
		vkcore.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.get_vk_physical_device(), surface.get_vk_surface(), &mut surface_properties)?;
		Ok(surface_properties.currentExtent)
	}

	/// Get the surface size, a.k.a. the frame buffer size
	pub fn get_surface_size(&self) -> Result<VkExtent2D, VulkanError> {
		Self::get_surface_size_(&self.vkcore, &self.device, self.surface.clone())
	}

	/// When the windows was resized, call this method to recreate the swapchain to fit the new size
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
}

