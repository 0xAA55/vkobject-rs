
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	mem::MaybeUninit,
	ptr::{null, null_mut},
	sync::Arc,
};

#[derive(Debug, Clone)]
pub struct VulkanGpuInfo {
	pub(crate) gpu: VkPhysicalDevice,
	pub(crate) properties: VkPhysicalDeviceProperties,
	pub(crate) mem_properties: VkPhysicalDeviceMemoryProperties,
	pub(crate) queue_families: Vec<VkQueueFamilyProperties>,
	pub(crate) extension_properties: Vec<VkExtensionProperties>,
}

impl VulkanGpuInfo {
	pub fn get_gpu_info(vkcore: &VkCore) -> Result<Vec<VulkanGpuInfo>, VulkanError> {
		let mut num_gpus = 0u32;
		vkcore.vkEnumeratePhysicalDevices(vkcore.get_instance(), &mut num_gpus, null_mut())?;
		let mut gpus = Vec::<VkPhysicalDevice>::with_capacity(num_gpus as usize);
		vkcore.vkEnumeratePhysicalDevices(vkcore.get_instance(), &mut num_gpus, gpus.as_mut_ptr())?;
		unsafe {gpus.set_len(num_gpus as usize)};
		let mut ret = Vec::<VulkanGpuInfo>::with_capacity(num_gpus as usize);
		for gpu in gpus {
			let mut properties: VkPhysicalDeviceProperties = unsafe {MaybeUninit::zeroed().assume_init()};
			vkcore.vkGetPhysicalDeviceProperties(gpu, &mut properties)?;
			let mut mem_properties: VkPhysicalDeviceMemoryProperties = unsafe {MaybeUninit::zeroed().assume_init()};
			vkcore.vkGetPhysicalDeviceMemoryProperties(gpu, &mut mem_properties)?;
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
				mem_properties,
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
	pub(crate) vkcore: Arc<VkCore>,
	queue_family_index: u32,
	gpu: VulkanGpuInfo,
	device: VkDevice,
	pub(crate) queues: Vec<Arc<Mutex<VkQueue>>>,
}

impl VulkanDevice {
	pub fn new(vkcore: Arc<VkCore>, gpu: VulkanGpuInfo, queue_family_index: u32) -> Result<Self, VulkanError> {
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

	/// Get the `VkPhysicalDevice`
	pub(crate) fn get_vk_physical_device(&self) -> VkPhysicalDevice {
		self.gpu.get_vk_physical_device()
	}

	/// Get the `VkDevice`
	pub(crate) fn get_vk_device(&self) -> VkDevice {
		self.device
	}

	pub fn get_vk_queue(&self) -> VkQueue {
		self.queue
	}

	pub fn get_supported_by_surface(&self, queue_index: usize, surface: VkSurfaceKHR) -> Result<bool, VulkanError> {
		let mut result: VkBool32 = 0;
		self.vkcore.vkGetPhysicalDeviceSurfaceSupportKHR(self.get_vk_physical_device(), queue_index as u32, surface, &mut result)?;
		Ok(result != 0)
	}

	pub fn wait_idle(&self) -> Result<(), VulkanError> {
		self.vkcore.vkDeviceWaitIdle(self.device)?;
		Ok(())
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
