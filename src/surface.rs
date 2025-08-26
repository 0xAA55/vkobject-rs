
use crate::prelude::*;
use std::{
	fmt::Debug,
	ptr::{null, null_mut},
	sync::{Mutex, Arc, Weak},
};

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
	fn new_from_ci<T>(function_name: &'static str, vkcore: &VkCore, device: &VulkanDevice, vk_create_surface: fn(VkInstance, &T, *const VkAllocationCallbacks, *mut VkSurfaceKHR) -> VkResult, surface_ci: &T) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let gpu_info = device.get_gpu();
		let mut surface: VkSurfaceKHR = null();
		vk_result_conv(function_name, vk_create_surface(vkcore.get_instance(), surface_ci, null(), &mut surface))?;

		let queue_families = gpu_info.get_queue_families();
		let mut supported = Vec::<bool>::with_capacity(queue_families.len());
		for i in 0..queue_families.len() {
			supported.push(device.get_supported_by_surface(i, surface)?);
		}
		let mut graphics_queue_node_index = u32::MAX;
		let mut present_queue_node_index = u32::MAX;
		for (i, queue_family) in queue_families.iter().enumerate() {
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
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, window: &glfw::PWindow) -> Result<Arc<Mutex<Self>>, VulkanError> {
		Self::new_from_ci("vkCreateWindowSurfaceGLFW", vkcore, device, vkCreateWindowSurfaceGLFW, window)
	}
	#[cfg(feature = "win32_khr")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, hwnd: HWND, hinstance: HINSTANCE) -> Result<Arc<Mutex<Self>>, VulkanError> {
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
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, window: *const ANativeWindow) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkAndroidSurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			window,
		};
		Self::new_from_ci("vkCreateAndroidSurfaceKHR", vkcore, device, vkCreateAndroidSurfaceKHR, &surface_ci)
	}
	#[cfg(feature = "ios_mvk")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, view: *const c_void) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkIOSSurfaceCreateInfoMVK {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IOS_SURFACE_CREATE_INFO_MVK,
			pNext: null(),
			flags: 0,
			pView: view,
		};
		Self::new_from_ci("vkCreateIOSSurfaceMVK", vkcore, device, vkCreateIOSSurfaceMVK, &surface_ci)
	}
	#[cfg(feature = "macos_mvk")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, view: *const c_void) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkMacOSSurfaceCreateInfoMVK {
			sType: VkStructureType::VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
			pNext: null(),
			flags: 0,
			pView: view,
		};
		Self::new_from_ci("vkCreateMacOSSurfaceMVK", vkcore, device, vkCreateMacOSSurfaceMVK, &surface_ci)
	}
	#[cfg(feature = "metal_ext")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, metal_layer: *const CAMetalLayer) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkMetalSurfaceCreateInfoEXT {
			sType: VkStructureType::VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
			pNext: null(),
			flags: 0,
			pLayer: metal_layer,
		};
		Self::new_from_ci("vkCreateMetalSurfaceEXT", vkcore, device, vkCreateMetalSurfaceEXT, &surface_ci)
	}
	#[cfg(feature = "wayland_khr")]
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, display: *const c_void, surface: *const c_void) -> Result<Arc<Mutex<Self>>, VulkanError> {
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
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, connection: *const c_void, window: xcb_window_t) -> Result<Arc<Mutex<Self>>, VulkanError> {
		let surface_ci = VkXcbSurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			connection,
			window,
		};
		Self::new_from_ci("vkCreateXcbSurfaceKHR", vkcore, device, vkCreateXcbSurfaceKHR, &surface_ci)
	}

	pub(crate) fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.ctx = ctx;
	}

	/// Get the current `VkSurfaceKHR`
	pub(crate) fn get_vk_surface(&self) -> VkSurfaceKHR {
		self.surface
	}

	/// Get the `VkSurfaceFormatKHR`
	pub(crate) fn get_vk_surface_format(&self) -> &VkSurfaceFormatKHR {
		&self.format
	}
}

impl Drop for VulkanSurface {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			vkcore.vkDestroySurfaceKHR(vkcore.get_instance(), self.surface, null()).unwrap();
		}
	}
}
