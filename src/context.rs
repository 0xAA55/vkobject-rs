
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
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

/// The policy of using V-Sync for the smoothest `present()` or the minimum-latency tearing `present()`.
#[derive(Debug, Clone, Copy)]
pub enum PresentInterval {
	/// Enable V-Sync for the smoothest `present()` with minimum power consumption.
	/// * If your device supports adaptive presentation interval, it will be automatically enabled.
	VSync,

	/// Disable V-Sync for maximum framerate with minimum-latency tearing `present()`.
	/// * The `usize` is the max concurrent frames; if the device doesn't limit the max concurrent frames, you have to provide one.
	/// * If you give zero, a default number of the max concurrent frames will be used.
	/// * The greater the number you give, the higher the framerate you could achieve, but the more memory will be used to store the framebuffer.
	MinLatencyPresent(usize),
}

impl PresentInterval {
	/// Is V-Sync requested?
	pub fn is_vsync(&self) -> bool {
		matches!(self, Self::VSync)
	}

	/// Is minimum-latency tearing `present()` requested?
	pub fn is_minimum_latency_present(&self) -> Option<usize> {
		if let Self::MinLatencyPresent(maximum_frames) = self {
			Some(*maximum_frames)
		} else {
			None
		}
	}
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

	/// V-Sync should be on or off?
	/// * It's recommended to enable V-Sync for most usage since this could be the smoothest achieve and lower the power consumption, **except** for players who play PVP and want to win with the lowest latency (You are designing these sorts of games)
	pub present_interval: PresentInterval,

	/// How many scenes could be rendered concurrently?
	/// **NOTE** You could create a multi-threaded rendering engine, recording draw commands concurrently, and the GPU could render multiple scenes concurrently.
	pub cpu_renderer_threads: usize,

	/// The size for the descriptor pool
	pub desc_pool_size: DescriptorPoolSize,

	/// Is this a VR project?
	pub is_vr: bool,
}

/// The Vulkan context has device, surface, swapchain, and command pools
#[derive(Debug)]
pub struct VulkanContext {
	/// The swapchain
	pub(crate) swapchain: Arc<VulkanSwapchain>,

	/// The pipeline cache here for a global usage
	pub pipeline_cache: Arc<VulkanPipelineCache>,

	/// The descriptor pool here is normally for a global usage
	pub desc_pool: Arc<DescriptorPool>,

	/// The command pools
	pub(crate) cmdpools: Vec<VulkanCommandPool>,

	/// The surface in use
	pub surface: Arc<VulkanSurface>,

	/// The device in use
	pub device: Arc<VulkanDevice>,

	/// The Vulkan driver
	pub(crate) vkcore: Arc<VkCore>,

	/// How many scenes could be rendered concurrently?
	pub cpu_renderer_threads: usize,
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

impl VulkanContext {
	/// Create a new `VulkanContext`
	pub fn new(create_info: VulkanContextCreateInfo) -> Result<Self, VulkanError> {
		let cpu_renderer_threads = create_info.cpu_renderer_threads;
		let vkcore = create_info.vkcore.clone();
		let device = Arc::new(match (create_info.device_can_graphics, create_info.device_can_compute) {
			(false, false) => VulkanDevice::choose_gpu_anyway(vkcore.clone())?,
			(true, false) => VulkanDevice::choose_gpu_with_graphics(vkcore.clone())?,
			(false, true) => VulkanDevice::choose_gpu_with_compute(vkcore.clone())?,
			(true, true) => VulkanDevice::choose_gpu_with_graphics_and_compute(vkcore.clone())?,
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

		let mut cmdpools: Vec<VulkanCommandPool> = Vec::with_capacity(cpu_renderer_threads);
		for _ in 0..cpu_renderer_threads {
			cmdpools.push(VulkanCommandPool::new(device.clone(), 2)?);
		}
		let desc_pool = Arc::new(DescriptorPool::new(device.clone(), create_info.desc_pool_size)?);
		let pipeline_cache = Arc::new(VulkanPipelineCache::new(device.clone(), Some(&load_cache("global_pipeline_cache", None).unwrap_or(Vec::new())))?);
		let size = Self::get_surface_size_(&vkcore, &device, &surface)?;
		let swapchain = Arc::new(VulkanSwapchain::new(device.clone(), surface.clone(), size.width, size.height, create_info.present_interval, cpu_renderer_threads, create_info.is_vr, None)?);
		let ret = Self {
			vkcore,
			device,
			surface,
			swapchain,
			cmdpools,
			desc_pool,
			pipeline_cache,
			cpu_renderer_threads,
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
	pub(crate) fn get_vk_queue(&self) -> VkQueue {
		self.device.get_vk_queue()
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

	/// Get the number of CPU renderer threads that supports
	pub fn get_supported_number_of_cpu_renderer_threads(&self) -> usize {
		self.cpu_renderer_threads
	}

	/// Recreate the swapchain when users toggle the switch of `vsync` or the framebuffer size changes
	pub fn recreate_swapchain(&mut self, width: u32, height: u32, present_interval: PresentInterval, is_vr: bool) -> Result<(), VulkanError> {
		self.device.wait_idle()?;
		self.swapchain = Arc::new(VulkanSwapchain::new(self.device.clone(), self.surface.clone(), width, height, present_interval, self.cpu_renderer_threads, is_vr, Some(self.swapchain.get_vk_swapchain()))?);
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
			let present_interval = self.swapchain.get_present_interval();
			let is_vr = self.swapchain.get_is_vr();
			self.recreate_swapchain(surface_size.width, surface_size.height, present_interval, is_vr)?;
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
			self.cmdpools[pool_index].use_pool(Some(rt_props))?
		} else {
			swapchain = Some(self.swapchain.clone());
			loop {
				match self.swapchain.acquire_next_image(pool_index, u64::MAX) {
					Ok(index) => {
						present_image_index = Some(index);
						break;
					}
					Err(e) => if let Some(ve) = e.is_vkerror() {
						match ve {
							VkError::VkErrorOutOfDateKhr(_) => {
								self.on_resize()?;
							}
							_ => return Err(VulkanError::VkError(ve.clone())),
						}
					} else {
						return Err(e)
					}
				};
			}
			self.cmdpools[pool_index].use_pool(Some(self.swapchain.get_image(present_image_index.unwrap()).rt_props.clone()))?
		};
		VulkanContextScene::new(self.device.vkcore.clone(), self.device.clone(), swapchain, pool_in_use, present_image_index)
	}
}

impl Drop for VulkanContext {
	fn drop(&mut self) {
		match self.pipeline_cache.dump_cache() {
			Ok(cache_data) => if let Err(reason) = save_cache("global_pipeline_cache", None, &cache_data) {
				eprintln!("Save pipeline cache data failed: {reason:?}");
			}
			Err(reason) => eprintln!("Dump pipeline cache data failed: {reason:?}"),
		}
	}
}

pub struct VulkanContextScene<'a> {
	pub vkcore: Arc<VkCore>,
	pub device: Arc<VulkanDevice>,
	pub pool_in_use: VulkanCommandPoolInUse<'a>,
	swapchain: Option<Arc<VulkanSwapchain>>,
	present_image_index: Option<usize>,
	present_queued: bool,
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
		for image in pool_in_use.rt_props.as_ref().unwrap().attachments.iter() {
			barrier.image = image.get_vk_image();
			barrier.subresourceRange.aspectMask = if image.is_depth_stencil() {
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as VkImageAspectFlags |
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as VkImageAspectFlags
			} else {
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags
			};
			vkcore.vkCmdPipelineBarrier(
				pool_in_use.cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT as VkPipelineStageFlags,
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
			present_queued: false,
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
		let extent = self.pool_in_use.rt_props.as_ref().unwrap().get_extent();
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
		self.set_scissor(*self.pool_in_use.rt_props.as_ref().unwrap().get_extent())
	}

	pub fn clear(&self, color: Vec4, depth: f32, stencil: u32) -> Result<(), VulkanError> {
		let cmdbuf = self.pool_in_use.cmdbuf;
		for image in self.pool_in_use.rt_props.as_ref().unwrap().attachments.iter() {
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
		if self.present_queued {
			panic!("Duplicated call to `VulkanContextScene::present()`.");
		}
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
		for image in self.pool_in_use.rt_props.as_ref().unwrap().attachments.iter() {
			barrier.image = image.get_vk_image();
			barrier.subresourceRange.aspectMask = if image.is_depth_stencil() {
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as VkImageAspectFlags |
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as VkImageAspectFlags
			} else {
				VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags
			};
			self.vkcore.vkCmdPipelineBarrier(
				self.pool_in_use.cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT as VkPipelineStageFlags,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				0,
				0, null(),
				0, null(),
				1, &barrier
			)?;
		}
		self.pool_in_use.submit().unwrap();
		match self.swapchain.as_ref().unwrap().queue_present(self.present_image_index.unwrap()) {
			Ok(_) => Ok(()),
			Err(e) => if let Some(ve) = e.is_vkerror() {
				match ve {
					VkError::VkErrorOutOfDateKhr(_) => Ok(()),
					_ => Err(VulkanError::VkError(ve.clone())),
				}
			} else {
				Err(e)
			}
		}?;
		self.present_queued = true;
		Ok(())
	}

	pub fn finish(self) {}
}

impl Debug for VulkanContextScene<'_> {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanContextScene")
		.field("pool_in_use", &self.pool_in_use)
		.field("swapchain", &self.swapchain)
		.field("present_image_index", &self.present_image_index)
		.field("present_queued", &self.present_queued)
		.finish()
	}
}

impl Drop for VulkanContextScene<'_> {
	fn drop(&mut self) {
		if self.present_image_index.is_some() {
			if !self.present_queued {
				self.present().unwrap();
			}
		} else {
			self.pool_in_use.submit().unwrap();
		}
	}
}
