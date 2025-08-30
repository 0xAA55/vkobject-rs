
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
	/// The Vulkan driver
	pub(crate) vkcore: Arc<VkCore>,

	/// The device in use
	pub device: Arc<VulkanDevice>,

	/// The surface in use
	pub surface: VulkanSurface,

	/// The swapchain
	pub(crate) swapchain: VulkanSwapchain,

	/// The command pools
	pub(crate) cmdpools: Vec<VulkanCommandPool>,
}

unsafe impl Send for VulkanContext {}

impl VulkanContext {
	/// Create a new `VulkanContext`
	pub fn new(create_info: VulkanContextCreateInfo) -> Result<Arc<Mutex<Self>>, VulkanError> {
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
		let surface = VulkanSurface::new(vkcore.clone(), &device, surface.window)?;
		#[cfg(feature = "win32_khr")]
		let surface = VulkanSurface::new(vkcore.clone(), &device, surface.hwnd, surface.hinstance)?;
		#[cfg(feature = "android_khr")]
		let surface = VulkanSurface::new(vkcore.clone(), &device, surface.window)?;
		#[cfg(feature = "ios_mvk")]
		let surface = VulkanSurface::new(vkcore.clone(), &device, surface.view)?;
		#[cfg(feature = "macos_mvk")]
		let surface = VulkanSurface::new(vkcore.clone(), &device, surface.view)?;
		#[cfg(feature = "metal_ext")]
		let surface = VulkanSurface::new(vkcore.clone(), &device, surface.metal_layer)?;
		#[cfg(feature = "wayland_khr")]
		let surface = VulkanSurface::new(vkcore.clone(), &device, surface.display, surface.surface)?;
		#[cfg(feature = "xcb_khr")]
		let surface = VulkanSurface::new(vkcore.clone(), &device, surface.connection, surface.window)?;

		let size = Self::get_surface_size_(&vkcore, &device, &surface)?;
		let ret = Arc::new(Mutex::new(Self {
			vkcore,
			device,
			surface,
			swapchain: None,
			cmdpools: Vec::with_capacity(max_concurrent_frames),
		}));
		let swapchain = VulkanSwapchain::new(ret.clone(), size.width, size.height, create_info.vsync, create_info.is_vr, None)?;
		let mut cmdpools: Vec<VulkanCommandPool> = Vec::with_capacity(max_concurrent_frames);
		for _ in 0..max_concurrent_frames {
			cmdpools.push(VulkanCommandPool::new(ret.clone(), 2)?);
		}
		let mut lock = ret.lock().unwrap();
		lock.swapchain = Some(swapchain);
		lock.cmdpools = cmdpools;
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

	/// Get an idle queue for the current device
	pub(crate) fn get_any_vk_queue(&self, queue_index: &mut usize) -> Result<MutexGuard<'_, VkQueue>, VulkanError> {
		self.device.get_any_vk_queue(queue_index)
	}

	/// Get an idle queue for the current device, will block if there's no idle queues
	pub(crate) fn get_any_vk_queue_anyway(&self, queue_index: &mut usize) -> MutexGuard<'_, VkQueue> {
		self.device.get_any_vk_queue_anyway(queue_index)
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
	pub fn get_swapchain_extent(&self) -> VkExtent2D {
		self.swapchain.get_swapchain_extent()
	}

	/// Get the swapchain image by an index
	pub fn get_swapchain_image(&self, index: usize) -> &VulkanSwapchainImage {
		self.swapchain.get_image(index)
	}

	/// Get the swapchain image by the current index
	pub fn get_cur_swapchain_image(&self) -> &VulkanSwapchainImage {
		self.swapchain.get_image(self.get_swapchain_image_index())
	}

	/// Get the swapchain
	pub fn get_swapchain(&self) -> &VulkanSwapchain {
		&self.swapchain
	}

	/// Get the current swapchain image index
	pub fn get_swapchain_image_index(&self) -> usize {
		self.swapchain.get_image_index() as usize
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

	/// Recreate the swapchain when users toggle the switch of `vsync` or the framebuffer size changes
	pub fn recreate_swapchain(&mut self, width: u32, height: u32, vsync: bool, is_vr: bool) -> Result<(), VulkanError> {
		self.device.wait_idle()?;
		self.swapchain = VulkanSwapchain::new(&self.vkcore, self.device.clone(), self.surface.clone(), width, height, vsync, is_vr, Some(self.get_vk_swapchain()))?;
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
			self.device.wait_idle()?;
			self.recreate_swapchain(surface_size.width, surface_size.height, self.swapchain.get_is_vsync(), self.swapchain.get_is_vr())?;
			Ok(true)
		}
	}

	/// Acquire a command buffer and a queue, start recording the commands
	/// * You could call this function in different threads, in order to achieve concurrent frame rendering
	pub fn begin_frame<'a>(&'a mut self, one_time_submit: bool) -> Result<VulkanContextFrame<'a>, VulkanError> {
		for (i, pool) in self.cmdpools.iter_mut().enumerate() {
			match pool.try_use_pool(Some(i), None, one_time_submit, None) {
				Ok(mut pool_in_use) => {
					let swapchain_image_index = self.swapchain.acquire_next_image()?;
					pool_in_use.swapchain_image_index = Some(swapchain_image_index);
					return Ok(VulkanContextFrame::new(pool_in_use, swapchain_image_index))
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
pub struct VulkanContextFrame<'a> {
	ctx: Arc<Mutex<VulkanContext>>,
	pool_in_use: VulkanCommandPoolInUse<'a>,
	swapchain_image_index: usize,
}

impl<'a> VulkanContextFrame<'a> {
	fn new(pool_in_use: VulkanCommandPoolInUse<'a>, swapchain_image_index: usize) -> Self {
		let ctx = pool_in_use.ctx.clone();
		Self {
			ctx,
			pool_in_use,
			swapchain_image_index,
		}
	}
}
