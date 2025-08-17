
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	mem::MaybeUninit,
	ptr::{null, null_mut},
	rc::Rc,
};

#[derive(Debug, Clone)]
pub enum VulkanError {
	VkError(VkError),
	ChooseGpuFailed,
}

impl From<VkError> for VulkanError {
	fn from(e: VkError) -> Self {
		Self::VkError(e)
	}
}

#[derive(Debug, Clone)]
pub struct VulkanGpuInfo {
	gpu: VkPhysicalDevice,
	queue_families: Vec<VkQueueFamilyProperties>,
	properties: VkPhysicalDeviceProperties,
}

impl VulkanGpuInfo {
	pub fn get_gpu_info(vkcore: Rc<VkCore>) -> Result<Vec<VulkanGpuInfo>, VkError> {
		let mut gpu_count = 0u32;
		vkcore.vkEnumeratePhysicalDevices(vkcore.instance, &mut gpu_count, null_mut())?;
		let mut gpus = Vec::<VkPhysicalDevice>::with_capacity(gpu_count as usize);
		vkcore.vkEnumeratePhysicalDevices(vkcore.instance, &mut gpu_count, gpus.as_mut_ptr())?;
		unsafe {gpus.set_len(gpu_count as usize)};
		let mut ret = Vec::<VulkanGpuInfo>::with_capacity(gpu_count as usize);
		for gpu in gpus {
			let mut queue_family_count = 0u32;
			vkcore.vkGetPhysicalDeviceQueueFamilyProperties(gpu, &mut queue_family_count, null_mut())?;
			let mut queue_families = Vec::<VkQueueFamilyProperties>::with_capacity(queue_family_count as usize);
			vkcore.vkGetPhysicalDeviceQueueFamilyProperties(gpu, &mut queue_family_count, queue_families.as_mut_ptr())?;
			unsafe {queue_families.set_len(queue_family_count as usize)};
			let mut properties: VkPhysicalDeviceProperties = unsafe {MaybeUninit::zeroed().assume_init()};
			vkcore.vkGetPhysicalDeviceProperties(gpu, &mut properties)?;
			ret.push(VulkanGpuInfo {
				gpu,
				queue_families,
				properties,
			});
		}
		Ok(ret)
	}

	pub fn get_gpu(&self) -> VkPhysicalDevice {
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
}

pub struct VulkanDevice {
	pub vkcore: Rc<VkCore>,
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
		let extensions = CStringArray::from_iter(vkcore.extensions.iter());
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
		vkcore.vkCreateDevice(gpu.get_gpu(), &device_create_info, null(), &mut device)?;

		Ok(Self {
			vkcore,
			gpu,
			device,
		})
	}

	pub fn get_gpu(&self) -> &VulkanGpuInfo {
		&self.gpu
	}

	pub fn get_device(&self) -> VkDevice {
		self.device
	}
}

impl Debug for VulkanDevice {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanDevice")
		.field("gpu", &self.gpu)
		.field("device", &self.device)
		.finish()
	}
}

impl Drop for VulkanDevice {
	fn drop(&mut self) {
		self.vkcore.vkDestroyDevice(self.device, null()).unwrap();
	}
}
