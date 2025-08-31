
use crate::prelude::*;
use std::{
	fmt::Debug,
	mem::MaybeUninit,
	sync::{Arc, Mutex, MutexGuard},
};

/// The struct to provide the information of the surface
#[derive(Debug)]
pub struct VulkanSurfaceInfo<'a> {
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
}

/// The struct to create the `VulkanContext`
#[derive(Debug)]
pub struct VulkanContextCreateInfo<'a> {
	/// The most important thing: the Vulkan driver is here
	pub vkcore: Arc<VkCore>,

	/// The device must support graphics?
	pub device_can_graphics: bool,

	/// The device must support compute?
	pub device_can_compute: bool,

	/// The surface, the target you want to render to
	pub surface: VulkanSurfaceInfo<'a>,

	/// VSYNC should be on or off?
	/// * It's recommended to enable VSYNC for most usage since this could be the smoothest achieve and lower the power consumption, **except** for players who play PVP and want to win with the lowest latency (You are designing these sorts of games)
	pub vsync: bool,

	/// How many frames could be rendered concurrently?
	/// **NOTE** You could create a multi-threaded rendering engine, submitting draw calls concurrently, and the GPU could render multiple scenes concurrently.
	pub max_concurrent_frames: usize,

	/// Is this a VR project?
	pub is_vr: bool,
}

/// The Vulkan context has device, surface, swapchain, and command pools
#[derive(Debug)]
pub struct VulkanContext {
	/// The swapchain
	pub(crate) swapchain: Arc<Mutex<VulkanSwapchain>>,

	/// The command pools
	pub(crate) cmdpools: Vec<VulkanCommandPool>,

	/// The surface in use
	pub surface: Arc<VulkanSurface>,

	/// The device in use
	pub device: Arc<VulkanDevice>,

	/// The Vulkan driver
	pub(crate) vkcore: Arc<VkCore>,
}

unsafe impl Send for VulkanContext {}

impl VulkanContext {
	/// Create a new `VulkanContext`
	pub fn new(create_info: VulkanContextCreateInfo) -> Result<Self, VulkanError> {
		let max_concurrent_frames = create_info.max_concurrent_frames;
		let vkcore = create_info.vkcore.clone();
		let device = Arc::new(match (create_info.device_can_graphics, create_info.device_can_compute) {
			(false, false) => VulkanDevice::choose_gpu_anyway(vkcore.clone(), max_concurrent_frames)?,
			(true, false) => VulkanDevice::choose_gpu_with_graphics(vkcore.clone(), max_concurrent_frames)?,
			(false, true) => VulkanDevice::choose_gpu_with_compute(vkcore.clone(), max_concurrent_frames)?,
			(true, true) => VulkanDevice::choose_gpu_with_graphics_and_compute(vkcore.clone(), max_concurrent_frames)?,
		});
		let surface = &create_info.surface;

		#[cfg(any(feature = "glfw", test))]
		let surface = Arc::new(VulkanSurface::new(vkcore.clone(), &device, surface.window)?);
		#[cfg(feature = "win32_khr")]
		let surface = Arc::new(VulkanSurface::new(vkcore.clone(), &device, surface.hwnd, surface.hinstance)?);
		#[cfg(feature = "android_khr")]
		let surface = Arc::new(VulkanSurface::new(vkcore.clone(), &device, surface.window)?);
		#[cfg(feature = "ios_mvk")]
		let surface = Arc::new(VulkanSurface::new(vkcore.clone(), &device, surface.view)?);
		#[cfg(feature = "macos_mvk")]
		let surface = Arc::new(VulkanSurface::new(vkcore.clone(), &device, surface.view)?);
		#[cfg(feature = "metal_ext")]
		let surface = Arc::new(VulkanSurface::new(vkcore.clone(), &device, surface.metal_layer)?);
		#[cfg(feature = "wayland_khr")]
		let surface = Arc::new(VulkanSurface::new(vkcore.clone(), &device, surface.display, surface.surface)?);
		#[cfg(feature = "xcb_khr")]
		let surface = Arc::new(VulkanSurface::new(vkcore.clone(), &device, surface.connection, surface.window)?);

		let size = Self::get_surface_size_(&vkcore, &device, &surface)?;
		let swapchain = Arc::new(Mutex::new(VulkanSwapchain::new(device.clone(), surface.clone(), size.width, size.height, create_info.vsync, create_info.is_vr, (max_concurrent_frames + 1) as u32, None)?));
		let mut cmdpools: Vec<VulkanCommandPool> = Vec::with_capacity(max_concurrent_frames);
		for _ in 0..max_concurrent_frames {
			cmdpools.push(VulkanCommandPool::new(device.clone(), 2)?);
		}
		let ret = Self {
			vkcore,
			device,
			surface,
			swapchain,
			cmdpools,
		};
		Ok(ret)
	}

	/// Get the Vulkan instance
	pub(crate) fn get_instance(&self) -> VkInstance {
		self.vkcore.get_instance()
	}

	/// get the `VkCore`
	pub(crate) fn get_vkcore(&self) -> Arc<VkCore> {
		self.vkcore.clone()
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
	pub(crate) fn get_vk_queue(&self, queue_index: usize) -> VkQueue {
		self.device.get_vk_queue(queue_index)
	}

	/// Get the `VkSurfaceKHR`
	pub(crate) fn get_vk_surface(&self) -> VkSurfaceKHR {
		self.surface.get_vk_surface()
	}

	/// Get the current surface format
	pub(crate) fn get_vk_surface_format(&self) -> &VkSurfaceFormatKHR {
		self.surface.get_vk_surface_format()
	}

	/// Get the swapchain
	pub fn get_swapchain<'a>(&'a self) -> MutexGuard<'a, VulkanSwapchain> {
		self.swapchain.lock().unwrap()
	}

	/// Get the `VkSwapchainKHR`
	pub(crate) fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.get_swapchain().get_vk_swapchain()
	}

	/// Get the current swapchain extent(the framebuffer size)
	pub fn get_swapchain_extent(&self) -> VkExtent2D {
		self.get_swapchain().get_swapchain_extent()
	}

	/// Get the surface size, a.k.a. the frame buffer size
	pub fn get_surface_size_(vkcore: &VkCore, device: &VulkanDevice, surface: &VulkanSurface) -> Result<VkExtent2D, VulkanError> {
		let mut surface_properties: VkSurfaceCapabilitiesKHR = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.get_vk_physical_device(), surface.get_vk_surface(), &mut surface_properties)?;
		Ok(surface_properties.currentExtent)
	}

	/// Get the surface size, a.k.a. the frame buffer size
	pub fn get_surface_size(&self) -> Result<VkExtent2D, VulkanError> {
		Self::get_surface_size_(&self.vkcore, &self.device, &self.surface)
	}

	/// Recreate the swapchain when users toggle the switch of `vsync` or the framebuffer size changes
	pub fn recreate_swapchain(&mut self, width: u32, height: u32, vsync: bool, is_vr: bool) -> Result<(), VulkanError> {
		self.device.wait_idle()?;
		let mut lock = self.swapchain.lock().unwrap();
		*lock = VulkanSwapchain::new(self.device.clone(), self.surface.clone(), width, height, vsync, is_vr, lock.get_desired_num_of_swapchain_images() as u32, Some(lock.get_vk_swapchain()))?;
		Ok(())
	}

	/// When the windows was resized, call this method to recreate the swapchain to fit the new size
	pub fn on_resize(&mut self) -> Result<bool, VulkanError> {
		let surface_size = self.get_surface_size()?;
		let swapchain_extent = self.get_swapchain_extent();
		if	swapchain_extent.width == surface_size.width &&
			swapchain_extent.height == surface_size.height {
			Ok(false)
		} else {
			let lock = self.swapchain.lock().unwrap();
			let is_vsync = lock.get_is_vsync();
			let is_vr = lock.get_is_vr();
			drop(lock);
			self.recreate_swapchain(surface_size.width, surface_size.height, is_vsync, is_vr)?;
			Ok(true)
		}
	}

	/// Acquire a command buffer and a queue, start recording the commands
	/// * You could call this function in different threads, in order to achieve concurrent frame rendering
	pub fn begin_frame(&mut self, one_time_submit: bool) -> Result<VulkanContextFrame, VulkanError> {
		for (i, pool) in self.cmdpools.iter_mut().enumerate() {
			match pool.try_use_pool(i, None, one_time_submit, None) {
				Ok(mut pool_in_use) => {
					let mut lock = self.swapchain.lock().unwrap();
					let image_index = lock.acquire_next_image(true)?;
					pool_in_use.swapchain_image = Some(lock.get_image(image_index));
					return Ok(VulkanContextFrame::new(self.swapchain.clone(), pool_in_use, image_index))
				}
				Err(e) => match e {
					VulkanError::CommandPoolIsInUse => {}
					others => return Err(others)
				}
			}
		}
		Err(VulkanError::NoIdleCommandPools)
	}
}

#[derive(Debug)]
pub struct VulkanContextFrame {
	swapchain: Arc<Mutex<VulkanSwapchain>>,
	pool_in_use: Option<VulkanCommandPoolInUse>,
	image_index: usize,
}

impl VulkanContextFrame {
	fn new(swapchain: Arc<Mutex<VulkanSwapchain>>, pool_in_use: VulkanCommandPoolInUse, image_index: usize) -> Self {
		Self {
			swapchain,
			pool_in_use: Some(pool_in_use),
			image_index,
		}
	}
}

impl Drop for VulkanContextFrame {
	fn drop(&mut self) {
		if let Some(ref mut pool_in_use) = self.pool_in_use {
			let queue_index = pool_in_use.queue_index;
			pool_in_use.submit().unwrap();
			let lock = self.swapchain.lock().unwrap();
			if let Err(e) = lock.queue_present(queue_index, self.image_index) {
				match e {
					VulkanError::VkError(e) => match e {
						VkError::VkErrorOutOfDateKhr(_) => {},
						other => Err(other).unwrap(),
					}
					other => Err(other).unwrap(),
				}
			}
		}
	}
}
