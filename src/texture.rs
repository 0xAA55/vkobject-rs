
use crate::prelude::*;
use std::{
	fmt::{self, Debug, Formatter},
	mem::MaybeUninit,
	ptr::null,
	sync::Arc,
};

/// The wrapper for the Vulkan texture images
pub struct VulkanTexture {
	/// The device holds all of the resource
	pub device: Arc<VulkanDevice>,

	/// The texture image
	image: VkImage,

	/// The width of the texture (pixels per row)
	width: u32,

	/// The height of the texture (rows per layer)
	height: u32,

	/// The depth of the texture (number of layers)
	depth: u32,

	/// The format of the texture
	format: VkFormat,

	/// The memory holds the image data
	memory: VulkanMemory,
}

impl VulkanTexture {
	/// Create the `VulkanTexture`
	pub fn new(device: Arc<VulkanDevice>, dim: VkImageType, width: u32, height: u32, depth: u32, format: VkFormat, tiling: VkImageTiling, flags: VkImageCreateFlags, usage: VkImageUsageFlags) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vkdevice = device.get_vk_device();
		let image_ci = VkImageCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			pNext: null(),
			flags,
			imageType: dim,
			format,
			extent: VkExtent3D {
				width,
				height,
				depth,
			},
			mipLevels: 1,
			arrayLayers: 1,
			samples: VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT,
			tiling,
			usage,
			sharingMode: VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
			queueFamilyIndexCount: 0,
			pQueueFamilyIndices: null(),
			initialLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
		};
		let mut image: VkImage = null();
		vkcore.vkCreateImage(vkdevice, &image_ci, null(), &mut image)?;
		let image = ResourceGuard::new(image, |&i|vkcore.clone().vkDestroyImage(vkdevice, i, null()).unwrap());
		let mut mem_reqs: VkMemoryRequirements = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetImageMemoryRequirements(vkdevice, *image, &mut mem_reqs)?;
		let memory = VulkanMemory::new(device.clone(), &mem_reqs, VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as u32)?;
		memory.bind_vk_image(*image)?;
		let image = image.release();
		Ok(Self {
			device,
			image,
			width,
			height,
			depth,
			format,
			memory,
		})
	}
}

impl Debug for VulkanTexture {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanTexture")
		.field("image", &self.image)
		.field("width", &self.width)
		.field("height", &self.height)
		.field("depth", &self.depth)
		.field("format", &self.format)
		.field("memory", &self.memory)
		.finish()
	}
}

impl Drop for VulkanTexture {
	fn drop(&mut self) {
		self.device.vkcore.vkDestroyImage(self.device.get_vk_device(), self.image, null()).unwrap();
	}
}
