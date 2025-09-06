
use crate::prelude::*;
use std::{
	fmt::Debug,
	mem::MaybeUninit,
	ptr::null,
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
	pub fn begin_frame<'a>(&'a mut self, pool_index: usize, one_time_submit: bool) -> Result<VulkanContextFrame<'a>, VulkanError> {
		let mut pool_in_use = self.cmdpools[pool_index].use_pool(pool_index, None, one_time_submit)?;
		let mut swapchain = self.swapchain.lock().unwrap();
		let present_image_index = swapchain.acquire_next_image(true)?;
		let swapchain_image = swapchain.get_image(present_image_index);
		pool_in_use.swapchain_image = Some(swapchain_image.clone());
		Ok(VulkanContextFrame::new(self.device.vkcore.clone(), self.device.clone(), self.swapchain.clone(), swapchain_image, pool_in_use, present_image_index)?)
	}
}

#[derive(Debug)]
pub struct VulkanContextFrame<'a> {
	pub vkcore: Arc<VkCore>,
	pub device: Arc<VulkanDevice>,
	pub swapchain: Arc<Mutex<VulkanSwapchain>>,
	pub swapchain_image: Arc<Mutex<VulkanSwapchainImage>>,
	pub barrier: VkImageMemoryBarrier,
	pub pool_in_use: VulkanCommandPoolInUse<'a>,
	present_image_index: usize,
}

impl<'a> VulkanContextFrame<'a> {
	fn new(vkcore: Arc<VkCore>, device: Arc<VulkanDevice>, swapchain: Arc<Mutex<VulkanSwapchain>>, swapchain_image: Arc<Mutex<VulkanSwapchainImage>>, pool_in_use: VulkanCommandPoolInUse<'a>, present_image_index: usize) -> Result<Self, VulkanError> {
		let barrier = VkImageMemoryBarrier {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			pNext: null(),
			srcAccessMask: 0,
			dstAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags,
			oldLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
			newLayout: VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
			image: swapchain_image.lock().unwrap().get_vk_image(),
			subresourceRange: VkImageSubresourceRange {
				aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
				baseMipLevel: 0,
				levelCount: 1,
				baseArrayLayer: 0,
				layerCount: 1,
			},
		};
		let ret = Self {
			vkcore,
			device,
			swapchain,
			swapchain_image,
			barrier,
			pool_in_use,
			present_image_index,
		};
		ret.vkcore.vkCmdPipelineBarrier(
			ret.pool_in_use.cmdbuf,
			VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT as VkPipelineStageFlags,
			VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
			0,
			0, null(),
			0, null(),
			1, &ret.barrier
		)?;
		Ok(ret)
	}

	pub fn get_present_image_index(&self) -> usize {
		self.present_image_index
	}

	pub fn set_viewport(&self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) -> Result<(), VulkanError> {
		let viewport = VkViewport {
			x,
			y,
			width,
			height,
			minDepth: min_depth,
			maxDepth: max_depth,
		};
		let cmdbuf = self.pool_in_use.cmdbuf;
		self.vkcore.vkCmdSetViewport(cmdbuf, 0, 1, &viewport)?;
		Ok(())
	}

	pub fn set_viewport_swapchain(&self, min_depth: f32, max_depth: f32) -> Result<(), VulkanError> {
		let extent = self.swapchain.lock().unwrap().get_swapchain_extent();
		self.set_viewport(0.0, 0.0, extent.width as f32, extent.height as f32, min_depth, max_depth)
	}

	pub fn set_scissor(&self, extent: VkExtent2D) -> Result<(), VulkanError> {
		let scissor = VkRect2D {
			offset: VkOffset2D {
				x: 0,
				y: 0,
			},
			extent,
		};
		self.vkcore.vkCmdSetScissor(self.pool_in_use.cmdbuf, 0, 1, &scissor)?;
		Ok(())
	}

	pub fn set_scissor_swapchain(&self) -> Result<(), VulkanError> {
		self.set_scissor(self.swapchain.lock().unwrap().get_swapchain_extent())
	}

	pub fn clear(&self, color: Vec4, depth: f32, stencil: u32) -> Result<(), VulkanError> {
		let cmdbuf = self.pool_in_use.cmdbuf;
		let lock = self.swapchain_image.lock().unwrap();
		let swapchain_image = lock.get_vk_image();
		let depth_stencil_image = lock.depth_stencil.get_vk_image();
		drop(lock);
		let color_clear_value = VkClearColorValue {
			float32: [color.x, color.y, color.z, color.w],
		};
		let depth_stencil_clear_value = VkClearDepthStencilValue {
			depth,
			stencil,
		};
		let range = VkImageSubresourceRange {
			aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
			baseMipLevel: 0,
			levelCount: 1,
			baseArrayLayer: 0,
			layerCount: 1,
		};
		self.vkcore.vkCmdClearColorImage(cmdbuf, swapchain_image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color_clear_value, 1, &range)?;
		self.vkcore.vkCmdClearDepthStencilImage(cmdbuf, depth_stencil_image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &depth_stencil_clear_value, 1, &range)?;
		Ok(())
	}
}

impl Drop for VulkanContextFrame<'_> {
	fn drop(&mut self) {
		self.barrier.oldLayout = VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		self.barrier.newLayout = VkImageLayout::VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		self.barrier.srcAccessMask = VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags;
		self.barrier.dstAccessMask = 0;
		self.vkcore.vkCmdPipelineBarrier(
			self.pool_in_use.cmdbuf,
			VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
			VkPipelineStageFlagBits::VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT as VkPipelineStageFlags,
			0,
			0, null(),
			0, null(),
			1, &self.barrier
		).unwrap();
		let queue_index = self.pool_in_use.queue_index;
		self.pool_in_use.submit().unwrap();
		let lock = self.swapchain.lock().unwrap();
		if let Err(e) = lock.queue_present(queue_index, self.present_image_index) {
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
