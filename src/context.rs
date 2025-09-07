
use crate::prelude::*;
use std::{
	fmt::Debug,
	mem::MaybeUninit,
	ptr::null,
	sync::Arc,
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
	pub(crate) swapchain: Arc<VulkanSwapchain>,

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
		let swapchain = Arc::new(VulkanSwapchain::new(device.clone(), surface.clone(), size.width, size.height, create_info.vsync, create_info.is_vr, (max_concurrent_frames + 1) as u32, None)?);
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
	pub fn get_swapchain(&self) -> Arc<VulkanSwapchain> {
		self.swapchain.clone()
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
		self.swapchain = Arc::new(VulkanSwapchain::new(self.device.clone(), self.surface.clone(), width, height, vsync, is_vr, self.swapchain.get_desired_num_of_swapchain_images() as u32, Some(self.swapchain.get_vk_swapchain()))?);
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
			let is_vsync = self.swapchain.get_is_vsync();
			let is_vr = self.swapchain.get_is_vr();
			self.recreate_swapchain(surface_size.width, surface_size.height, is_vsync, is_vr)?;
			Ok(true)
		}
	}

	/// Acquire a command buffer and a queue, start recording the commands
	/// * You could call this function in different threads, in order to achieve concurrent frame rendering
	pub fn begin_scene<'a>(&'a mut self, pool_index: usize, rt_props: Option<Arc<RenderTargetProps>>) -> Result<VulkanContextScene<'a>, VulkanError> {
		let present_image_index;
		let swapchain;
		let pool_in_use = if let Some(rt_props) = rt_props {
			swapchain = None;
			present_image_index = None;
			self.cmdpools[pool_index].use_pool(pool_index, rt_props)?
		} else {
			swapchain = Some(self.swapchain.clone());
			let index = self.swapchain.acquire_next_image(true)?;
			present_image_index = Some(index);
			self.cmdpools[pool_index].use_pool(pool_index, self.swapchain.get_image(index).rt_props.clone())?
		};
		Ok(VulkanContextScene::new(self.device.vkcore.clone(), self.device.clone(), swapchain, pool_in_use, present_image_index)?)
	}
}

#[derive(Debug)]
pub struct VulkanContextScene<'a> {
	pub vkcore: Arc<VkCore>,
	pub device: Arc<VulkanDevice>,
	pub pool_in_use: VulkanCommandPoolInUse<'a>,
	swapchain: Option<Arc<VulkanSwapchain>>,
	present_image_index: Option<usize>,
}

impl<'a> VulkanContextScene<'a> {
	pub(crate) fn new(vkcore: Arc<VkCore>, device: Arc<VulkanDevice>, swapchain: Option<Arc<VulkanSwapchain>>, pool_in_use: VulkanCommandPoolInUse<'a>, present_image_index: Option<usize>) -> Result<Self, VulkanError> {
		let mut barrier = VkImageMemoryBarrier {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			pNext: null(),
			srcAccessMask: 0,
			dstAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags,
			oldLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
			newLayout: VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
			image: null(),
			subresourceRange: VkImageSubresourceRange {
				aspectMask: 0,
				baseMipLevel: 0,
				levelCount: 1,
				baseArrayLayer: 0,
				layerCount: 1,
			},
		};
		for image in pool_in_use.rt_props.attachments.iter() {
			barrier.image = image.get_vk_image();
			barrier.subresourceRange.aspectMask = if image.is_depth_stencil() {
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as VkImageAspectFlags |
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as VkImageAspectFlags
			} else {
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags
			};
			vkcore.vkCmdPipelineBarrier(
				pool_in_use.cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT as VkPipelineStageFlags,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				0,
				0, null(),
				0, null(),
				1, &barrier
			)?;
		}
		Ok(Self {
			vkcore,
			device,
			pool_in_use,
			swapchain,
			present_image_index,
		})
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
		let extent = self.pool_in_use.get_extent();
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
		self.set_scissor(self.pool_in_use.get_extent())
	}

	pub fn clear(&self, color: Vec4, depth: f32, stencil: u32) -> Result<(), VulkanError> {
		let cmdbuf = self.pool_in_use.cmdbuf;
		for image in self.pool_in_use.rt_props.attachments.iter() {
			if !image.is_depth_stencil() {
				let color_clear_value = VkClearColorValue {
					float32: [color.x, color.y, color.z, color.w],
				};
				let range = VkImageSubresourceRange {
					aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
					baseMipLevel: 0,
					levelCount: 1,
					baseArrayLayer: 0,
					layerCount: 1,
				};
				self.vkcore.vkCmdClearColorImage(cmdbuf, image.get_vk_image(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color_clear_value, 1, &range)?;
			} else {
				let depth_stencil_clear_value = VkClearDepthStencilValue {
					depth,
					stencil,
				};
				let range = VkImageSubresourceRange {
					aspectMask: 
						VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as VkImageAspectFlags |
						VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as VkImageAspectFlags,
					baseMipLevel: 0,
					levelCount: 1,
					baseArrayLayer: 0,
					layerCount: 1,
				};
				self.vkcore.vkCmdClearDepthStencilImage(cmdbuf, image.get_vk_image(), VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &depth_stencil_clear_value, 1, &range)?;
			}
		}
		Ok(())
	}

	pub fn present(&mut self) -> Result<(), VulkanError> {
		let queue_index = self.pool_in_use.queue_index;
		let mut barrier = VkImageMemoryBarrier {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			pNext: null(),
			srcAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags,
			dstAccessMask: 0,
			oldLayout: VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			newLayout: VkImageLayout::VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
			image: null(),
			subresourceRange: VkImageSubresourceRange {
				aspectMask: 0,
				baseMipLevel: 0,
				levelCount: 1,
				baseArrayLayer: 0,
				layerCount: 1,
			},
		};
		for image in self.pool_in_use.rt_props.attachments.iter() {
			barrier.image = image.get_vk_image();
			barrier.subresourceRange.aspectMask = if image.is_depth_stencil() {
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as VkImageAspectFlags |
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as VkImageAspectFlags
			} else {
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags
			};
			self.vkcore.vkCmdPipelineBarrier(
				self.pool_in_use.cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT as VkPipelineStageFlags,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				0,
				0, null(),
				0, null(),
				1, &barrier
			)?;
		}
		self.pool_in_use.submit().unwrap();
		self.swapchain.as_ref().unwrap().queue_present(queue_index, self.present_image_index.unwrap())?;
		Ok(())
	}

	pub fn finish(self) {}
}

impl Drop for VulkanContextScene<'_> {
	fn drop(&mut self) {
		if let Some(_) = self.present_image_index {
			self.present().unwrap();
		} else {
			self.pool_in_use.submit().unwrap();
		}
	}
}
