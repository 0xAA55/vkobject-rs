
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	mem::MaybeUninit,
	ptr::{null, null_mut},
	sync::Arc,
};

/// The physical device info
#[derive(Debug, Clone)]
pub struct VulkanGpuInfo {
	/// The physical device
	pub(crate) gpu: VkPhysicalDevice,

	/// The properties of the physical device
	pub(crate) properties: VkPhysicalDeviceProperties,

	/// The properties of the physical device memory
	pub(crate) mem_properties: VkPhysicalDeviceMemoryProperties,

	/// The queue families
	pub(crate) queue_families: Vec<VkQueueFamilyProperties>,

	/// The extension properties
	pub(crate) extension_properties: Vec<VkExtensionProperties>,
}

impl VulkanGpuInfo {
	/// Create a list of all the GPUs info
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

	/// Get the `VkPhysicalDevice`
	pub(crate) fn get_vk_physical_device(&self) -> VkPhysicalDevice {
		self.gpu
	}

	/// Get the `VkQueueFamilyProperties` list
	pub fn get_queue_families(&self) -> &[VkQueueFamilyProperties] {
		self.queue_families.as_ref()
	}

	/// Find a queue family index that matches the flags
	pub fn get_queue_family_index_by_flags(&self, queue_flag_match: u32) -> u32 {
		for i in 0..self.queue_families.len() {
			if (self.queue_families[i].queueFlags & queue_flag_match) == queue_flag_match {
				return i as u32;
			}
		}
		u32::MAX
	}

	/// Get the `VkPhysicalDeviceProperties`
	pub fn get_properties(&self) -> &VkPhysicalDeviceProperties {
		&self.properties
	}

	/// Get the list of the `VkExtensionProperties`
	pub fn get_extension_properties(&self) -> &[VkExtensionProperties] {
		self.extension_properties.as_ref()
	}

	/// Get memory type index by the memory properties flags
	pub fn get_memory_type_index(&self, mut type_bits: u32, properties: VkMemoryPropertyFlags) -> Result<u32, VulkanError> {
		for i in 0..self.mem_properties.memoryTypeCount {
			if (type_bits & 1) == 1 {
				if (self.mem_properties.memoryTypes[i as usize].propertyFlags & properties) == properties {
					return Ok(i)
				}
			}
			type_bits >>= 1;
		}
		Err(VulkanError::NoSuitableMemoryType)
	}
}

unsafe impl Send for VulkanGpuInfo {}
unsafe impl Sync for VulkanGpuInfo {}

/// The Vulkan device with its queues to submit the rendering commands
pub struct VulkanDevice {
	/// The Vulkan driver
	pub(crate) vkcore: Arc<VkCore>,

	/// The current queue family index
	queue_family_index: u32,

	/// The info of the GPU
	gpu: VulkanGpuInfo,

	/// The handle to the device
	device: VkDevice,

	/// The queues of the device. Submit commands to the queue to control GPU.
	pub(crate) queues: Vec<VkQueue>,
}

impl VulkanDevice {
	/// Create the `VulkanDevice` by the given `VkPhysicalDevice` and the queue family index
	/// * `queue_count`: **important**: This argument determines how many concurrent rendering tasks are allowed in the GPU.
	///   * Too much causes huge system memory/video memory usage,
	///     while too little causes the GPU to be unable to run multiple drawing command queues at once.
	pub fn new(vkcore: Arc<VkCore>, gpu: VulkanGpuInfo, queue_family_index: u32, queue_count: usize) -> Result<Self, VulkanError> {
		let priorities = [1.0];
		let queue_ci = VkDeviceQueueCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			queueFamilyIndex: queue_family_index,
			queueCount: queue_count as u32,
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
		let device = ResourceGuard::new(device, |&d|vkcore.vkDestroyDevice(d, null()).unwrap());

		let mut queues: Vec<VkQueue> = Vec::with_capacity(queue_count);
		for i in 0..queue_count {
			let mut queue: VkQueue = null();
			vkcore.vkGetDeviceQueue(*device, queue_family_index, i, &mut queue)?;
			queues.push(queue);
		}

		Ok(Self {
			vkcore: vkcore.clone(),
			queue_family_index,
			gpu,
			device: device.release(),
			queues,
		})
	}

	/// Choose a GPU that matches the `VkQueueFlags`
	/// * `flags`: The flags you want to match
	/// * `queue_count`: see `VulkanDevice::new()`
	pub fn choose_gpu(vkcore: Arc<VkCore>, flags: VkQueueFlags, queue_count: usize) -> Result<Self, VulkanError> {
		for gpu in VulkanGpuInfo::get_gpu_info(&vkcore)?.iter() {
			let index = gpu.get_queue_family_index_by_flags(flags);
			if index != u32::MAX {
				return Self::new(vkcore, gpu.clone(), index, queue_count);
			}
		}
		Err(VulkanError::ChooseGpuFailed)
	}

	/// Choose a GPU that must support graphics
	/// * `queue_count`: see `VulkanDevice::new()`
	pub fn choose_gpu_with_graphics(vkcore: Arc<VkCore>, queue_count: usize) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore, VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT as u32, queue_count)
	}

	/// Choose a GPU that must support compute
	/// * `queue_count`: see `VulkanDevice::new()`
	pub fn choose_gpu_with_compute(vkcore: Arc<VkCore>, queue_count: usize) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore, VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT as u32, queue_count)
	}

	/// Choose a GPU that must support both graphics and compute
	/// * `queue_count`: see `VulkanDevice::new()`
	pub fn choose_gpu_with_graphics_and_compute(vkcore: Arc<VkCore>, queue_count: usize) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore,
			VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT as u32 |
			VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT as u32,
			queue_count)
	}

	/// Choose a GPU that is, anyway, a GPU regardless of can do graphics/compute or not.
	/// * `queue_count`: see `VulkanDevice::new()`
	pub fn choose_gpu_anyway(vkcore: Arc<VkCore>, queue_count: usize) -> Result<Self, VulkanError> {
		Self::choose_gpu(vkcore, 0, queue_count)
	}

	/// Get the current queue family index
	pub fn get_queue_family_index(&self) -> u32 {
		self.queue_family_index
	}

	/// Get the GPU info
	pub fn get_gpu(&self) -> &VulkanGpuInfo {
		&self.gpu
	}

	/// Check if the `queue_index` and the `VkSurfaceKHR` were supported by the `VkPhysicalDevice`
	pub fn get_supported_by_surface(&self, queue_index: usize, surface: VkSurfaceKHR) -> Result<bool, VulkanError> {
		let mut result: VkBool32 = 0;
		self.vkcore.vkGetPhysicalDeviceSurfaceSupportKHR(self.get_vk_physical_device(), queue_index as u32, surface, &mut result)?;
		Ok(result != 0)
	}

	/// A wrapper for `vkDeviceWaitIdle`
	pub fn wait_idle(&self) -> Result<(), VulkanError> {
		self.vkcore.vkDeviceWaitIdle(self.device)?;
		Ok(())
	}

	/// Get the `VkPhysicalDevice`
	pub(crate) fn get_vk_physical_device(&self) -> VkPhysicalDevice {
		self.gpu.get_vk_physical_device()
	}

	/// Get the `VkDevice`
	pub(crate) fn get_vk_device(&self) -> VkDevice {
		self.device
	}

	/// Get a queue for the current device
	pub(crate) fn get_vk_queue(&self, queue_index: usize) -> VkQueue {
		self.queues[queue_index]
	}
}

impl Debug for VulkanDevice {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanDevice")
		.field("queue_family_index", &self.queue_family_index)
		.field("gpu", &self.gpu)
		.field("device", &self.device)
		.field("queues", &self.queues)
		.finish()
	}
}

impl Clone for VulkanDevice {
	fn clone(&self) -> Self {
		Self::new(self.vkcore.clone(), self.gpu.clone(), self.queue_family_index, self.queues.len()).unwrap()
	}
}

impl Drop for VulkanDevice {
	fn drop(&mut self) {
		self.vkcore.vkDestroyDevice(self.device, null()).unwrap();
	}
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}
