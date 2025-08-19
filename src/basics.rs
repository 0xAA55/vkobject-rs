
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::MaybeUninit,
	ptr::{null, null_mut},
	rc::Rc,
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
	pub vkcore: Rc<VkCore>,
	queue_family_index: u32,
	gpu: VulkanGpuInfo,
	device: VkDevice,
}

impl VulkanDevice {
	pub fn new(vkcore: Rc<VkCore>, gpu: VulkanGpuInfo, queue_family_index: u32) -> Result<Self, VkError> {
		let priorities = vec![1.0];
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

	pub fn choose_gpu(vkcore: Rc<VkCore>, flags: VkQueueFlags) -> Result<Self, VulkanError> {
		for gpu in VulkanGpuInfo::get_gpu_info(&vkcore)?.iter() {
			let index = gpu.get_queue_family_index(flags);
			if index != u32::MAX {
				return Ok(Self::new(vkcore, gpu.clone(), index)?);
			}
		}
		Err(VulkanError::ChooseGpuFailed)
	}

	pub fn choose_gpu_with_graphics(vkcore: Rc<VkCore>) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore, VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT as u32)
	}

	pub fn choose_gpu_with_compute(vkcore: Rc<VkCore>) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore, VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT as u32)
	}

	pub fn choose_gpu_with_graphics_and_compute(vkcore: Rc<VkCore>) -> Result<Self, VulkanError> {
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
	pub states: Weak<VulkanStates>,
	surface: VkSurfaceKHR,
	format: VkSurfaceFormatKHR,
}

impl VulkanSurface {
	pub fn new_from(states: Rc<VulkanStates>, surface: VkSurfaceKHR, format: VkSurfaceFormatKHR) -> Self {
		Self {
			states: Rc::downgrade(&states),
			surface,
			format,
		}
	}
	#[allow(dead_code)]
	fn new_from_ci<T>(function_name: &'static str, states: Rc<VulkanStates>, vk_create_surface: fn(VkInstance, &T, *const VkAllocationCallbacks, *mut VkSurfaceKHR) -> VkResult, surface_ci: &T) -> Result<Self, VulkanError> {
		let vkcore = &states.vkcore;
		let device = &states.device;
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
		Ok(Self::new_from(states, surface, selected_format))
	}
	#[cfg(any(feature = "glfw", test))]
	pub fn new(states: Rc<VulkanStates>, window: &glfw::PWindow) -> Result<Self, VkError> {
		let vkcore = &states.vkcore;
		let mut surface: VkSurfaceKHR = null();
		Self::new_from_ci("vkCreateWindowSurfaceGLFW", states, vkCreateWindowSurfaceGLFW, window)
	}
	#[cfg(feature = "win32_khr")]
	pub fn new(states: Rc<VulkanStates>, wnd: HWND, hinstance: HINSTANCE) -> Result<Self, VkError> {
		let surface_ci = VkWin32SurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			hinstance,
			hwnd,
		};
		Self::new_from_ci("vkCreateWin32SurfaceKHR", states, vkCreateWin32SurfaceKHR, &surface_ci)
	}
	#[cfg(feature = "android_khr")]
	pub fn new(states: Rc<VulkanStates>, window: *const ANativeWindow) -> Result<Self, VkError> {
		let surface_ci = VkAndroidSurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			window,
		};
		Self::new_from_ci("vkCreateAndroidSurfaceKHR", states, vkCreateAndroidSurfaceKHR, &surface_ci)
	}
	#[cfg(feature = "ios_mvk")]
	pub fn new(states: Rc<VulkanStates>, view: *const c_void) -> Result<Self, VkError> {
		let surface_ci = VkIOSSurfaceCreateInfoMVK {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IOS_SURFACE_CREATE_INFO_MVK,
			pNext: null(),
			flags: 0,
			pView: view,
		};
		Self::new_from_ci("vkCreateIOSSurfaceMVK", states, vkCreateIOSSurfaceMVK, &surface_ci)
	}
	#[cfg(feature = "macos_mvk")]
	pub fn new(states: Rc<VulkanStates>, view: *const c_void) -> Result<Self, VkError> {
		let surface_ci = VkMacOSSurfaceCreateInfoMVK {
			sType: VkStructureType::VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
			pNext: null(),
			flags: 0,
			pView: view,
		};
		Self::new_from_ci("vkCreateMacOSSurfaceMVK", states, vkCreateMacOSSurfaceMVK, &surface_ci)
	}
	#[cfg(feature = "metal_ext")]
	pub fn new(states: Rc<VulkanStates>, metal_layer: *const CAMetalLayer) -> Result<Self, VkError> {
		let surface_ci = VkMetalSurfaceCreateInfoEXT {
			sType: VkStructureType::VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
			pNext: null(),
			flags: 0,
			pLayer: metal_layer,
		};
		Self::new_from_ci("vkCreateMetalSurfaceEXT", states, vkCreateMetalSurfaceEXT, &surface_ci)
	}
	#[cfg(feature = "wayland_khr")]
	pub fn new(states: Rc<VulkanStates>, display: *const c_void, surface: *const c_void) -> Result<Self, VkError> {
		let surface_ci = VkWaylandSurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			display,
			surface,
		};
		Self::new_from_ci("vkCreateWaylandSurfaceKHR", states, vkCreateWaylandSurfaceKHR, &surface_ci)
	}
	#[cfg(feature = "xcb_khr")]
	pub fn new(states: Rc<VulkanStates>, connection: *const c_void, window: xcb_window_t) -> Result<Self, VkError> {
		let surface_ci = VkXcbSurfaceCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			connection,
			window,
		};
		Self::new_from_ci("vkCreateXcbSurfaceKHR", states, vkCreateXcbSurfaceKHR, &surface_ci)
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
		let states = self.states.upgrade().unwrap();
		let vkcore = &states.vkcore;
		vkcore.vkDestroySurfaceKHR(vkcore.instance, self.surface, null()).unwrap();
	}
}

