
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
	pub vkcore: Arc<VkCore>,
	queue_family_index: u32,
	gpu: VulkanGpuInfo,
	device: VkDevice,
}

impl VulkanDevice {
	pub fn new(vkcore: Arc<VkCore>, gpu: VulkanGpuInfo, queue_family_index: u32) -> Result<Self, VkError> {
		let priorities = [1.0];
		let queue_create_info = VkDeviceQueueCreateInfo {
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
		let device_create_info = VkDeviceCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			queueCreateInfoCount: 1,
			pQueueCreateInfos: &queue_create_info as *const _,
			enabledLayerCount: 0,
			ppEnabledLayerNames: null(),
			enabledExtensionCount: extensions.len() as u32,
			ppEnabledExtensionNames: extensions.as_ptr(),
			pEnabledFeatures: null(),
		};

		let mut device: VkDevice = null();
		vkcore.vkCreateDevice(gpu.get_vk_physical_device(), &device_create_info, null(), &mut device)?;

		Ok(Self {
			vkcore,
			queue_family_index,
			gpu,
			device,
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

#[allow(dead_code)]
fn vk_check(function_name: &'static str, result: VkResult) -> Result<(), VkError> {
	match result {
		VkResult::VK_SUCCESS => Ok(()),
		others => vk_convert_result(function_name, Ok(others)),
	}
}

#[derive(Debug)]
pub struct VulkanSurface {
	pub states: Weak<Mutex<VulkanStates>>,
	surface: VkSurfaceKHR,
	format: VkSurfaceFormatKHR,
}

unsafe impl Send for VulkanSurface {}

impl VulkanSurface {
	pub fn new_from(surface: VkSurfaceKHR, format: VkSurfaceFormatKHR) -> Arc<Mutex<Self>> {
		Arc::new(Mutex::new(Self {
			states: Weak::new(),
			surface,
			format,
		}))
	}
	#[allow(dead_code)]
	fn new_from_ci<T>(function_name: &'static str, vkcore: &VkCore, device: &VulkanDevice,  vk_create_surface: fn(VkInstance, &T, *const VkAllocationCallbacks, *mut VkSurfaceKHR) -> VkResult, surface_ci: &T) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let gpu_info = &device.gpu;
		let mut surface: VkSurfaceKHR = null();
		vk_check(function_name, vk_create_surface(vkcore.instance, surface_ci, null(), &mut surface))?;

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

		'find_format: for pf in PREFERRED_FORMAT.iter() {
			for f in formats.iter() {
				if f.format == *pf {
					selected_format = *f;
					break 'find_format;
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

	pub fn get_vk_surface(&self) -> VkSurfaceKHR {
		self.surface
	}

	pub fn get_vk_surface_format(&self) -> &VkSurfaceFormatKHR {
		&self.format
	}
}

impl Drop for VulkanSurface {
	fn drop(&mut self) {
		if let Some(binding) = self.states.upgrade() {
			let states = binding.lock().unwrap();
			let vkcore = &states.vkcore;
			vkcore.vkDestroySurfaceKHR(vkcore.instance, self.surface, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanSwapchain {
	pub states: Weak<Mutex<VulkanStates>>,
	pub surface: Weak<Mutex<VulkanSurface>>,
	swapchain: VkSwapchainKHR,
	swapchain_extent: VkExtent2D,
	present_mode: VkPresentModeKHR,
	images: Vec<VkImage>,
	image_views: Vec<VkImageView>,
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
				if *mode == VkPresentModeKHR::VK_PRESENT_MODE_MAILBOX_KHR {
					present_mode = *mode;
					break;
				} else if *mode == VkPresentModeKHR::VK_PRESENT_MODE_IMMEDIATE_KHR {
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
		let pre_transform: VkSurfaceTransformFlagBitsKHR = unsafe {transmute(if (surf_caps.supportedTransforms as u32 &
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
			if (surf_caps.supportedCompositeAlpha as u32 & *flag as u32) == *flag as u32 {
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
				(surf_caps.supportedUsageFlags as u32 & VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_SRC_BIT as u32) |
				(surf_caps.supportedUsageFlags as u32 & VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT as u32)
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
		let mut images = Vec::<VkImage>::with_capacity(num_images as usize);
		vkcore.vkGetSwapchainImagesKHR(vk_device, swapchain, &mut num_images, images.as_mut_ptr())?;
		unsafe {images.set_len(num_images as usize)};
		let mut image_views = Vec::<VkImageView>::with_capacity(images.len());
		for image in images.iter() {
			let vk_image_view_ci = VkImageViewCreateInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				pNext: null(),
				flags: 0,
				image: *image,
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
			vkcore.vkCreateImageView(vk_device, &vk_image_view_ci, null(), &mut image_view)?;
			image_views.push(image_view);
		}

		Ok(Self {
			states: Weak::new(),
			surface: Arc::downgrade(&surface_arc),
			swapchain,
			swapchain_extent,
			present_mode,
			images,
			image_views,
		})
	}

	pub fn get_vk_surface(&self) -> VkSurfaceKHR {
		let binding = self.surface.upgrade().unwrap();
		let surface = binding.lock().unwrap();
		surface.get_vk_surface()
	}

	pub fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.swapchain
	}

	pub fn get_swapchain_extent(&self) -> VkExtent2D {
		self.swapchain_extent
	}

	pub fn get_present_mode(&self) -> VkPresentModeKHR {
		self.present_mode
	}

	pub fn get_vk_images(&self) -> &[VkImage] {
		self.images.as_ref()
	}

	pub fn get_vk_image_views(&self) -> &[VkImageView] {
		self.image_views.as_ref()
	}

	pub fn acquire_next_image(&self, states: &VulkanStates, present_complete_semaphore: VkSemaphore, image_index: &mut u32) -> Result<(), VkError> {
		let vkcore = &states.vkcore;
		let device = states.get_vk_device();
		vkcore.vkAcquireNextImageKHR(device, self.swapchain, u64::MAX, present_complete_semaphore, null(), image_index)
	}

	pub fn queue_present(&self, states: &VulkanStates, queue: VkQueue, image_index: u32, wait_semaphore: VkSemaphore) -> Result<(), VkError> {
		let vkcore = &states.vkcore;
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
		vkcore.vkQueuePresentKHR(queue, &present_info)
	}
}

impl Drop for VulkanSwapchain {
	fn drop(&mut self) {
		if let Some(binding) = self.states.upgrade() {
			let states = binding.lock().unwrap();
			let vkcore = &states.vkcore;
			let device = states.get_vk_device();
			for image_view in self.image_views.iter() {
				vkcore.vkDestroyImageView(device, *image_view, null()).unwrap();
			}
			vkcore.vkDestroySwapchainKHR(states.get_vk_device(), self.swapchain, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanCommandPool {
	pub states: Weak<Mutex<VulkanStates>>,
	pool: VkCommandPool,
	cmd_buffers: Vec<VkCommandBuffer>,
	fences: Vec<VkFence>,
}

unsafe impl Send for VulkanCommandPool {}

impl VulkanCommandPool {
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, max_concurrent_frames: usize) -> Result<Self, VkError> {
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
			commandBufferCount: max_concurrent_frames as u32,
		};
		let mut cmd_buffers = Vec::<VkCommandBuffer>::with_capacity(max_concurrent_frames);
		vkcore.vkAllocateCommandBuffers(vk_device, &cmd_buffers_ci, cmd_buffers.as_mut_ptr())?;
		unsafe {cmd_buffers.set_len(max_concurrent_frames)};
		let mut fences = Vec::<VkFence>::with_capacity(max_concurrent_frames);
		unsafe {fences.set_len(max_concurrent_frames)};
		for fence in fences.iter_mut() {
			let fence_ci = VkFenceCreateInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
				pNext: null(),
				flags: VkFenceCreateFlagBits::VK_FENCE_CREATE_SIGNALED_BIT as u32,
			};
			vkcore.vkCreateFence(vk_device, &fence_ci, null(), fence)?;
		}
		Ok(Self{
			states: Weak::new(),
			pool,
			cmd_buffers,
			fences,
		})
	}

	/// Retrieve the command pool
	pub fn get_pool(&self) -> VkCommandPool {
		self.pool
	}

	/// Get the command buffers
	pub fn get_cmd_buffers(&self) -> &[VkCommandBuffer] {
		self.cmd_buffers.as_ref()
	}

	/// Get the command buffers as mutable reference
	pub fn get_cmd_buffers_mut(&mut self) -> &mut [VkCommandBuffer] {
		self.cmd_buffers.as_mut()
	}

	/// Get the fences
	pub fn get_fences(&self) -> &[VkFence] {
		self.fences.as_ref()
	}
}

impl Drop for VulkanCommandPool {
	fn drop(&mut self) {
		if let Some(binding) = self.states.upgrade() {
			let states = binding.lock().unwrap();
			let vkcore = &states.vkcore;
			let device = states.get_vk_device();
			for fence in self.fences.iter() {
				vkcore.vkDestroyFence(device, *fence, null()).unwrap();
			}
			vkcore.vkDestroyCommandPool(states.get_vk_device(), self.pool, null()).unwrap();
		}
	}
}

#[derive(Debug, Clone)]
pub struct VulkanStates {
	pub vkcore: Arc<VkCore>,
	pub device: Arc<VulkanDevice>,
	pub surface: Arc<Mutex<VulkanSurface>>,
	pub swapchain: Arc<Mutex<VulkanSwapchain>>,
	pub cmdpool: Arc<Mutex<VulkanCommandPool>>,
}

unsafe impl Send for VulkanStates {}

impl VulkanStates {
	/// Create a new `VulkanStates`
	pub fn new(vkcore: Arc<VkCore>, device: Arc<VulkanDevice>, surface: Arc<Mutex<VulkanSurface>>, width: u32, height: u32, vsync: bool, max_concurrent_frames: usize, is_vr: bool) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let ret = Arc::new(Mutex::new(Self{
			vkcore: vkcore.clone(),
			device: device.clone(),
			surface: surface.clone(),
			swapchain: Arc::new(Mutex::new(VulkanSwapchain::new(&vkcore, &device, surface.clone(), width, height, vsync, is_vr)?)),
			cmdpool: Arc::new(Mutex::new(VulkanCommandPool::new(&vkcore, &device, max_concurrent_frames)?)),
		}));
		let weak = Arc::downgrade(&ret);
		if true {
			let borrow = ret.lock().unwrap();
			borrow.surface.lock().unwrap().states = weak.clone();
			borrow.swapchain.lock().unwrap().states = weak.clone();
			borrow.cmdpool.lock().unwrap().states = weak.clone();
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
