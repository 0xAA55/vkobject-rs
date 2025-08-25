
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::{MaybeUninit, transmute},
	ptr::{null, null_mut},
	sync::{Mutex, Arc, Weak},
};

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
