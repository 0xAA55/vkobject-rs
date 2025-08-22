
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

pub struct VulkanDevice {
	vkcore: Arc<VkCore>,
	queue_family_index: u32,
	gpu: VulkanGpuInfo,
	device: VkDevice,
	queue: VkQueue,
}

impl VulkanDevice {
	pub fn new(vkcore: Arc<VkCore>, gpu: VulkanGpuInfo, queue_family_index: u32) -> Result<Self, VkError> {
		let priorities = [1.0];
		let queue_ci = VkDeviceQueueCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			queueFamilyIndex: queue_family_index,
			queueCount: 1,
			pQueuePriorities: priorities.as_ptr(),
		};
		let mut extensions = Vec::<*const i8>::with_capacity(gpu.extension_properties.len());
		for ext in gpu.extension_properties.iter() {
			extensions.push(&ext.extensionName[0] as *const _);
		}
		let device_ci = VkDeviceCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			queueCreateInfoCount: 1,
			pQueueCreateInfos: &queue_ci as *const _,
			enabledLayerCount: 0,
			ppEnabledLayerNames: null(),
			enabledExtensionCount: extensions.len() as u32,
			ppEnabledExtensionNames: extensions.as_ptr(),
			pEnabledFeatures: null(),
		};

		let mut device: VkDevice = null();
		vkcore.vkCreateDevice(gpu.get_vk_physical_device(), &device_ci, null(), &mut device)?;

		let mut queue: VkQueue = null();
		vkcore.vkGetDeviceQueue(device, queue_family_index, 0, &mut queue)?;

		Ok(Self {
			vkcore,
			queue_family_index,
			gpu,
			device,
			queue,
		})
	}

	pub fn choose_gpu(vkcore: Arc<VkCore>, flags: VkQueueFlags) -> Result<Self, VulkanError> {
		for gpu in VulkanGpuInfo::get_gpu_info(&vkcore)?.iter() {
			let index = gpu.get_queue_family_index(flags);
			if index != u32::MAX {
				return Ok(Self::new(vkcore, gpu.clone(), index)?);
			}
		}
		Err(VulkanError::ChooseGpuFailed)
	}

	pub fn choose_gpu_with_graphics(vkcore: Arc<VkCore>) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore, VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT as u32)
	}

	pub fn choose_gpu_with_compute(vkcore: Arc<VkCore>) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore, VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT as u32)
	}

	pub fn choose_gpu_with_graphics_and_compute(vkcore: Arc<VkCore>) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore,
			VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT as u32 |
			VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT as u32)
	}

	pub fn get_queue_family_index(&self) -> u32 {
		self.queue_family_index
	}

	pub fn get_gpu(&self) -> &VulkanGpuInfo {
		&self.gpu
	}

	pub fn get_vk_physical_device(&self) -> VkPhysicalDevice {
		self.gpu.get_vk_physical_device()
	}

	pub fn get_vk_device(&self) -> VkDevice {
		self.device
	}

	pub fn get_vk_queue(&self) -> VkQueue {
		self.queue
	}

	pub fn get_supported_by_surface(&self, queue_index: usize, surface: VkSurfaceKHR) -> Result<bool, VkError> {
		let mut result: VkBool32 = 0;
		self.vkcore.vkGetPhysicalDeviceSurfaceSupportKHR(self.get_vk_physical_device(), queue_index as u32, surface, &mut result)?;
		Ok(result != 0)
	}
}

impl Debug for VulkanDevice {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanDevice")
		.field("queue_family_index", &self.queue_family_index)
		.field("gpu", &self.gpu)
		.field("device", &self.device)
		.finish()
	}
}

impl Clone for VulkanDevice {
	fn clone(&self) -> Self {
		Self::new(self.vkcore.clone(), self.gpu.clone(), self.queue_family_index).unwrap()
	}
}

impl Drop for VulkanDevice {
	fn drop(&mut self) {
		self.vkcore.vkDestroyDevice(self.device, null()).unwrap();
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
			let vkcore = &ctx.vkcore;
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
			let vkcore = &ctx.vkcore;
			vkcore.vkDestroyFence(ctx.get_vk_device(), self.fence, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanSurface {
	ctx: Weak<Mutex<VulkanContext>>,
	surface: VkSurfaceKHR,
	format: VkSurfaceFormatKHR,
}

unsafe impl Send for VulkanSurface {}

impl VulkanSurface {
	pub fn new_from(surface: VkSurfaceKHR, format: VkSurfaceFormatKHR) -> Arc<Mutex<Self>> {
		Arc::new(Mutex::new(Self {
			ctx: Weak::new(),
			surface,
			format,
		}))
	}
	#[allow(dead_code)]
	fn new_from_ci<T>(function_name: &'static str, vkcore: &VkCore, device: &VulkanDevice,  vk_create_surface: fn(VkInstance, &T, *const VkAllocationCallbacks, *mut VkSurfaceKHR) -> VkResult, surface_ci: &T) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let gpu_info = &device.gpu;
		let mut surface: VkSurfaceKHR = null();
		vk_result_conv(function_name, vk_create_surface(vkcore.instance, surface_ci, null(), &mut surface))?;

		let mut supported = Vec::<bool>::with_capacity(gpu_info.queue_families.len());
		for i in 0..gpu_info.queue_families.len() {
			supported.push(device.get_supported_by_surface(i, surface)?);
		}
		let mut graphics_queue_node_index = u32::MAX;
		let mut present_queue_node_index = u32::MAX;
		for (i, queue_family) in gpu_info.queue_families.iter().enumerate() {
			if (queue_family.queueFlags & VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT as u32) == VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT as u32 {
				graphics_queue_node_index = i as u32;
				if supported[i] {
					present_queue_node_index = i as u32;
					break;
				}
			}
		}
		if present_queue_node_index == u32::MAX {
			for (i, s) in supported.iter().enumerate() {
				if *s {
					present_queue_node_index = i as u32;
					break;
				}
			}
		}
		if graphics_queue_node_index == u32::MAX && present_queue_node_index == u32::MAX {
			return Err(VulkanError::NoGoodQueueForSurface("Could not find a graphics and/or presenting queue!"));
		}
		if graphics_queue_node_index != present_queue_node_index {
			return Err(VulkanError::NoGoodQueueForSurface("Separate graphics and presenting queues are not supported yet!"));
		}
		let mut num_formats: u32 = 0;
		vkcore.vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_info.get_vk_physical_device(), surface, &mut num_formats, null_mut())?;
		let mut formats = Vec::<VkSurfaceFormatKHR>::with_capacity(num_formats as usize);
		vkcore.vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_info.get_vk_physical_device(), surface, &mut num_formats, formats.as_mut_ptr())?;
		unsafe {formats.set_len(num_formats as usize)};

		let mut selected_format = formats[0];

		const PREFERRED_FORMAT: [VkFormat; 3] = [
			VkFormat::VK_FORMAT_B8G8R8A8_UNORM,
			VkFormat::VK_FORMAT_R8G8B8A8_UNORM,
			VkFormat::VK_FORMAT_A8B8G8R8_UNORM_PACK32,
		];

		'find_format: {
			for pf in PREFERRED_FORMAT.iter() {
				for f in formats.iter() {
					if f.format == *pf {
						selected_format = *f;
						break 'find_format;
					}
				}
			}
		}
		Ok(Self::new_from(surface, selected_format))
	}
	#[cfg(any(feature = "glfw", test))]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice,  window: &glfw::PWindow) -> Result<Arc<Mutex<Self>>, VulkanError> {
		Self::new_from_ci("vkCreateWindowSurfaceGLFW", vkcore, device, vkCreateWindowSurfaceGLFW, window)
	}
	#[cfg(feature = "win32_khr")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice,  wnd: HWND, hinstance: HINSTANCE) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkWin32SurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			hinstance,
			hwnd,
		};
		Self::new_from_ci("vkCreateWin32SurfaceKHR", vkcore, device, vkCreateWin32SurfaceKHR, &surface_ci)
	}
	#[cfg(feature = "android_khr")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice,  window: *const ANativeWindow) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkAndroidSurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			window,
		};
		Self::new_from_ci("vkCreateAndroidSurfaceKHR", vkcore, device, vkCreateAndroidSurfaceKHR, &surface_ci)
	}
	#[cfg(feature = "ios_mvk")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice,  view: *const c_void) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkIOSSurfaceCreateInfoMVK {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IOS_SURFACE_CREATE_INFO_MVK,
			pNext: null(),
			flags: 0,
			pView: view,
		};
		Self::new_from_ci("vkCreateIOSSurfaceMVK", vkcore, device, vkCreateIOSSurfaceMVK, &surface_ci)
	}
	#[cfg(feature = "macos_mvk")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice,  view: *const c_void) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkMacOSSurfaceCreateInfoMVK {
			sType: VkStructureType::VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
			pNext: null(),
			flags: 0,
			pView: view,
		};
		Self::new_from_ci("vkCreateMacOSSurfaceMVK", vkcore, device, vkCreateMacOSSurfaceMVK, &surface_ci)
	}
	#[cfg(feature = "metal_ext")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice,  metal_layer: *const CAMetalLayer) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkMetalSurfaceCreateInfoEXT {
			sType: VkStructureType::VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
			pNext: null(),
			flags: 0,
			pLayer: metal_layer,
		};
		Self::new_from_ci("vkCreateMetalSurfaceEXT", vkcore, device, vkCreateMetalSurfaceEXT, &surface_ci)
	}
	#[cfg(feature = "wayland_khr")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice,  display: *const c_void, surface: *const c_void) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkWaylandSurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			display,
			surface,
		};
		Self::new_from_ci("vkCreateWaylandSurfaceKHR", vkcore, device, vkCreateWaylandSurfaceKHR, &surface_ci)
	}
	#[cfg(feature = "xcb_khr")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice,  connection: *const c_void, window: xcb_window_t) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkXcbSurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			connection,
			window,
		};
		Self::new_from_ci("vkCreateXcbSurfaceKHR", vkcore, device, vkCreateXcbSurfaceKHR, &surface_ci)
	}

	/// Get the current `VkSurfaceKHR`
	pub fn get_vk_surface(&self) -> VkSurfaceKHR {
		self.surface
	}

	/// Get the `VkSurfaceFormatKHR`
	pub fn get_vk_surface_format(&self) -> &VkSurfaceFormatKHR {
		&self.format
	}
}

impl Drop for VulkanSurface {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = &ctx.vkcore;
			vkcore.vkDestroySurfaceKHR(vkcore.instance, self.surface, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanSwapchainImage {
	ctx: Weak<Mutex<VulkanContext>>,
	image: VkImage,
	image_view: VkImageView,
	acquire_semaphore: VulkanSemaphore,
	release_semaphore: VulkanSemaphore,
	queue_submit_fence: VulkanFence,
}

unsafe impl Send for VulkanSwapchainImage {}

impl VulkanSwapchainImage {
	pub fn new(vkcore: &VkCore, image: VkImage, surface: &VulkanSurface, device: &VulkanDevice) -> Result<Self, VkError> {
		let vk_image_view_ci = VkImageViewCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			pNext: null(),
			flags: 0,
			image,
			viewType: VkImageViewType::VK_IMAGE_VIEW_TYPE_2D,
			format: surface.format.format,
			components: VkComponentMapping {
				r: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_R,
				g: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_G,
				b: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_B,
				a: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_A,
			},
			subresourceRange: VkImageSubresourceRange {
				aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as u32,
				baseMipLevel: 0,
				levelCount: 1,
				baseArrayLayer: 0,
				layerCount: 1,
			},
		};
		let mut image_view: VkImageView = null();
		let acquire_semaphore = VulkanSemaphore::new(vkcore, device)?;
		let release_semaphore = VulkanSemaphore::new(vkcore, device)?;
		let queue_submit_fence = VulkanFence::new(vkcore, device)?;
		let vk_device = device.get_vk_device();
		vkcore.vkCreateImageView(vk_device, &vk_image_view_ci, null(), &mut image_view)?;
		Ok(Self{
			ctx: Weak::new(),
			image,
			image_view,
			acquire_semaphore,
			release_semaphore,
			queue_submit_fence,
		})
	}

	pub fn get_vk_image(&self) -> VkImage {
		self.image
	}

	fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.acquire_semaphore.set_ctx(ctx.clone());
		self.release_semaphore.set_ctx(ctx.clone());
		self.queue_submit_fence.set_ctx(ctx.clone());
		self.ctx = ctx;
	}
}

impl Drop for VulkanSwapchainImage {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = &ctx.vkcore;
			let device = ctx.get_vk_device();
			vkcore.vkDestroyImageView(device, self.image_view, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanSwapchain {
	ctx: Weak<Mutex<VulkanContext>>,
	pub surface: Weak<Mutex<VulkanSurface>>,
	surf_caps: VkSurfaceCapabilitiesKHR,
	swapchain: VkSwapchainKHR,
	swapchain_extent: VkExtent2D,
	present_mode: VkPresentModeKHR,
	images: Vec<VulkanSwapchainImage>,
}

unsafe impl Send for VulkanSwapchain {}

impl VulkanSwapchain {
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, surface_arc: Arc<Mutex<VulkanSurface>>, width: u32, height: u32, vsync: bool, is_vr: bool) -> Result<Self, VkError> {
		let surface = surface_arc.lock().unwrap();
		let vk_device = device.get_vk_device();
		let vk_phy_dev = device.get_vk_physical_device();
		let vk_surface = surface.get_vk_surface();

		let mut surf_caps: VkSurfaceCapabilitiesKHR  = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk_phy_dev, vk_surface, &mut surf_caps)?;

		let swapchain_extent = if surf_caps.currentExtent.width == u32::MAX {
			VkExtent2D {
				width,
				height,
			}
		} else {
			surf_caps.currentExtent
		};

		let mut num_present_mode = 0u32;
		vkcore.vkGetPhysicalDeviceSurfacePresentModesKHR(vk_phy_dev, vk_surface, &mut num_present_mode, null_mut())?;
		let mut present_modes = Vec::<VkPresentModeKHR>::with_capacity(num_present_mode as usize);
		vkcore.vkGetPhysicalDeviceSurfacePresentModesKHR(vk_phy_dev, vk_surface, &mut num_present_mode, present_modes.as_mut_ptr())?;
		unsafe {present_modes.set_len(num_present_mode as usize)};

		// Select a present mode for the swapchain
		let mut present_mode = VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR;

		// If v-sync is not requested, try to find a mailbox mode
		// It's the lowest latency non-tearing present mode available
		if !vsync {
			for mode in present_modes.iter() {
				if *mode == VkPresentModeKHR::VK_PRESENT_MODE_MAILBOX_KHR || *mode == VkPresentModeKHR::VK_PRESENT_MODE_IMMEDIATE_KHR {
					present_mode = *mode;
					break;
				}
			}
		}

		// Determine the number of images
		let mut desired_num_of_swapchain_images = surf_caps.minImageCount + 1;
		if surf_caps.maxImageCount > 0 && desired_num_of_swapchain_images > surf_caps.maxImageCount {
			desired_num_of_swapchain_images = surf_caps.maxImageCount;
		}

		// Find the transformation of the surface
		let pre_transform: VkSurfaceTransformFlagBitsKHR = unsafe {transmute(if (surf_caps.supportedTransforms &
			VkSurfaceTransformFlagBitsKHR::VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR as u32) ==
			VkSurfaceTransformFlagBitsKHR::VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR as u32 {
			VkSurfaceTransformFlagBitsKHR::VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
		} else {
			surf_caps.currentTransform
		} as u32)};

		// Find a supported composite alpha format (not all devices support alpha opaque)
		let mut composite_alpha = VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		const COMPOSITE_ALPHA_FLAGS: [VkCompositeAlphaFlagBitsKHR; 4] = [
			VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
			VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
			VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
		];
		for flag in COMPOSITE_ALPHA_FLAGS.iter() {
			if (surf_caps.supportedCompositeAlpha & *flag as u32) == *flag as u32 {
				composite_alpha = *flag;
				break;
			}
		}

		let swapchain_ci = VkSwapchainCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			surface: vk_surface,
			minImageCount: desired_num_of_swapchain_images,
			imageFormat: surface.format.format,
			imageColorSpace: surface.format.colorSpace,
			imageExtent: swapchain_extent,
			imageArrayLayers: if !is_vr {1} else {2},
			imageUsage: VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT as u32 |
				(surf_caps.supportedUsageFlags & VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_SRC_BIT as u32) |
				(surf_caps.supportedUsageFlags & VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT as u32)
				as VkImageUsageFlags,
			imageSharingMode: VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
			queueFamilyIndexCount: 0,
			pQueueFamilyIndices: null(),
			preTransform: pre_transform,
			compositeAlpha: composite_alpha,
			presentMode: present_mode,
			clipped: VK_TRUE,
			oldSwapchain: null(),
		};

		let mut swapchain: VkSwapchainKHR = null();
		vkcore.vkCreateSwapchainKHR(vk_device, &swapchain_ci, null(), &mut swapchain)?;
		let mut num_images = 0u32;
		vkcore.vkGetSwapchainImagesKHR(vk_device, swapchain, &mut num_images, null_mut())?;
		let mut vk_images = Vec::<VkImage>::with_capacity(num_images as usize);
		vkcore.vkGetSwapchainImagesKHR(vk_device, swapchain, &mut num_images, vk_images.as_mut_ptr())?;
		unsafe {vk_images.set_len(num_images as usize)};
		let mut images = Vec::<VulkanSwapchainImage>::with_capacity(vk_images.len());
		for vk_image in vk_images.iter() {
			images.push(VulkanSwapchainImage::new(vkcore, *vk_image, &surface, device)?);
		}

		Ok(Self {
			ctx: Weak::new(),
			surface: Arc::downgrade(&surface_arc),
			surf_caps,
			swapchain,
			swapchain_extent,
			present_mode,
			images,
		})
	}

	fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		for image in self.images.iter_mut() {
			image.set_ctx(ctx.clone());
		}
		self.ctx = ctx;
	}

	pub fn get_vk_surface(&self) -> VkSurfaceKHR {
		let binding = self.surface.upgrade().unwrap();
		let surface = binding.lock().unwrap();
		surface.get_vk_surface()
	}

	pub fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.swapchain
	}

	pub fn get_vk_surf_caps(&self) -> &VkSurfaceCapabilitiesKHR {
		&self.surf_caps
	}

	pub fn get_swapchain_extent(&self) -> VkExtent2D {
		self.swapchain_extent
	}

	pub fn get_present_mode(&self) -> VkPresentModeKHR {
		self.present_mode
	}

	pub fn get_images(&self) -> &[VulkanSwapchainImage] {
		self.images.as_ref()
	}

	pub fn acquire_next_image(&self, present_complete_semaphore: VkSemaphore, image_index: &mut u32) -> Result<(), VkError> {
		let binding = self.ctx.upgrade().unwrap();
		let ctx = binding.lock().unwrap();
		let vkcore = &ctx.vkcore;
		let device = ctx.get_vk_device();
		vkcore.vkAcquireNextImageKHR(device, self.swapchain, u64::MAX, present_complete_semaphore, null(), image_index)?;
		Ok(())
	}

	pub fn queue_present(&self, image_index: u32, wait_semaphore: VkSemaphore) -> Result<(), VkError> {
		let binding = self.ctx.upgrade().unwrap();
		let ctx = binding.lock().unwrap();
		let vkcore = &ctx.vkcore;
		let num_wait_semaphores;
		let wait_semaphores;
		if wait_semaphore != VK_NULL_HANDLE as _ {
			num_wait_semaphores = 1;
			wait_semaphores = &wait_semaphore as *const _;
		} else {
			num_wait_semaphores = 0;
			wait_semaphores = null();
		}
		let present_info = VkPresentInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			pNext: null(),
			waitSemaphoreCount: num_wait_semaphores,
			pWaitSemaphores: wait_semaphores,
			swapchainCount: 1,
			pSwapchains: &self.swapchain as *const _,
			pImageIndices: &image_index as *const _,
			pResults: null_mut(),
		};
		vkcore.vkQueuePresentKHR(ctx.get_vk_queue(), &present_info)?;
		Ok(())
	}
}

impl Drop for VulkanSwapchain {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = &ctx.vkcore;
			self.images.clear();
			vkcore.vkDestroySwapchainKHR(ctx.get_vk_device(), self.swapchain, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanCommandPool {
	ctx: Weak<Mutex<VulkanContext>>,
	pool: VkCommandPool,
	cmd_buffer: VkCommandBuffer,
	fence: VkFence,
}

unsafe impl Send for VulkanCommandPool {}

impl VulkanCommandPool {
	pub fn new(vkcore: &VkCore, device: &VulkanDevice) -> Result<Self, VkError> {
		let vk_device = device.get_vk_device();
		let pool_ci = VkCommandPoolCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			pNext: null(),
			queueFamilyIndex: device.queue_family_index,
			flags: VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT as u32,
		};
		let mut pool: VkCommandPool = null();
		vkcore.vkCreateCommandPool(vk_device, &pool_ci, null(), &mut pool)?;
		let cmd_buffers_ci = VkCommandBufferAllocateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			pNext: null(),
			commandPool: pool,
			level: VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			commandBufferCount: 1,
		};
		let mut cmd_buffer: VkCommandBuffer = null();
		vkcore.vkAllocateCommandBuffers(vk_device, &cmd_buffers_ci, &mut cmd_buffer)?;
		let mut fence: VkFence = null();
		let fence_ci = VkFenceCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			pNext: null(),
			flags: VkFenceCreateFlagBits::VK_FENCE_CREATE_SIGNALED_BIT as u32,
		};
		vkcore.vkCreateFence(vk_device, &fence_ci, null(), &mut fence)?;
		Ok(Self{
			ctx: Weak::new(),
			pool,
			cmd_buffer,
			fence,
		})
	}

	/// Retrieve the command pool
	pub fn get_vk_cmdpool(&self) -> VkCommandPool {
		self.pool
	}

	/// Get the command buffers
	pub fn get_vk_cmd_buffer(&self) -> VkCommandBuffer {
		self.cmd_buffer
	}

	/// Get the fences
	pub fn get_vk_fence(&self) -> VkFence {
		self.fence
	}
}

impl Drop for VulkanCommandPool {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = &ctx.vkcore;
			let device = ctx.get_vk_device();
			vkcore.vkDestroyFence(device, self.fence, null()).unwrap();
			vkcore.vkDestroyCommandPool(ctx.get_vk_device(), self.pool, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanCommandPoolInUse<'a> {
	ctx: Arc<Mutex<VulkanContext>>,
	cmdpool: &'a VulkanCommandPool,
	image_index: usize,
}

impl<'a> VulkanCommandPoolInUse<'a> {
	pub fn new(cmdpool: &'a VulkanCommandPool, image_index: usize) -> Result<Self, VkError> {
		let ctx = cmdpool.ctx.upgrade().unwrap();
		let ctx_g = ctx.lock().unwrap();
		let vkcore = &ctx_g.vkcore;
		let cmdbuf = cmdpool.get_vk_cmd_buffer();
		let begin_info = VkCommandBufferBeginInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			pNext: null(),
			flags: VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT as u32,
			pInheritanceInfo: null(),
		};
		vkcore.vkBeginCommandBuffer(cmdbuf, &begin_info)?;
		Ok(Self {
			ctx: ctx.clone(),
			cmdpool,
			image_index,
		})
	}

	pub fn submit(self) {}
}

impl Drop for VulkanCommandPoolInUse<'_> {
	fn drop(&mut self) {
		let ctx = self.ctx.lock().unwrap();
		let vkcore = &ctx.vkcore;
		let cmdbuf = self.cmdpool.get_vk_cmd_buffer();
		vkcore.vkEndCommandBuffer(cmdbuf).unwrap();

		let swapchain = ctx.swapchain.lock().unwrap();
		let images = swapchain.get_images();

		let wait_stage = [VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT as VkPipelineStageFlags];
		let cmd_buffers = [cmdbuf];
		let submit_info = VkSubmitInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SUBMIT_INFO,
			pNext: null(),
			waitSemaphoreCount: 1,
			pWaitSemaphores: &images[self.image_index].acquire_semaphore.get_vk_semaphore(),
			pWaitDstStageMask: wait_stage.as_ptr(),
			commandBufferCount: 1,
			pCommandBuffers: cmd_buffers.as_ptr(),
			signalSemaphoreCount: 1,
			pSignalSemaphores: &images[self.image_index].release_semaphore.get_vk_semaphore(),
		};
		vkcore.vkQueueSubmit(ctx.get_vk_queue(), 1, &submit_info, images[self.image_index].queue_submit_fence.get_vk_fence()).unwrap();
	}
}

#[derive(Debug, Clone)]
pub struct VulkanContext {
	vkcore: Arc<VkCore>,
	pub device: Arc<VulkanDevice>,
	pub surface: Arc<Mutex<VulkanSurface>>,
	pub swapchain: Arc<Mutex<VulkanSwapchain>>,
	pub cmdpools: Vec<Arc<Mutex<VulkanCommandPool>>>,
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
		}));
		let weak = Arc::downgrade(&ret);
		if true {
			let mut borrow = ret.lock().unwrap();
			borrow.surface.lock().unwrap().ctx = weak.clone();
			borrow.swapchain.lock().unwrap().set_ctx(weak.clone());
			for cmdpool in borrow.cmdpools.iter_mut() {
				cmdpool.lock().unwrap().ctx = weak.clone();
			}
		}
		Ok(ret)
	}

	/// Get the Vulkan instance
	pub fn get_instance(&self) -> VkInstance {
		self.vkcore.instance
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
}

