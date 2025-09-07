
use crate::prelude::*;
use std::{
	cmp::max,
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::MaybeUninit,
	ptr::null,
	sync::Arc,
};

/// The texture type and size
#[derive(Debug, Clone, Copy)]
pub enum VulkanTextureType {
	T1d(u32),
	T2d(VkExtent2D),
	T3d(VkExtent3D),
	Cube(u32),
	DepthStencil(VkExtent2D),
}

impl VulkanTextureType {
	/// Get if the image is cubemap
	pub fn is_cube(&self) -> bool {
		if let Self::Cube(_) = self {
			true
		} else {
			false
		}
	}

	/// Get if the image is depth stencil
	pub fn is_depth_stencil(&self) -> bool {
		if let Self::DepthStencil(_) = self {
			true
		} else {
			false
		}
	}

	/// Get the `VkImageType`
	pub fn get_image_type(&self) -> VkImageType {
		match self {
			Self::T1d(_) => {
				VkImageType::VK_IMAGE_TYPE_1D
			}
			Self::T2d(_) => {
				VkImageType::VK_IMAGE_TYPE_2D
			}
			Self::T3d(_) => {
				VkImageType::VK_IMAGE_TYPE_3D
			}
			Self::Cube(_) => {
				VkImageType::VK_IMAGE_TYPE_2D
			}
			Self::DepthStencil(_) => {
				VkImageType::VK_IMAGE_TYPE_2D
			}
		}
	}

	/// Get the `VkExtent3D`
	pub fn get_extent(&self) -> VkExtent3D {
		match self {
			Self::T1d(size) => {
				VkExtent3D {
					width: *size,
					height: 1,
					depth: 1,
				}
			}
			Self::T2d(size) => {
				VkExtent3D {
					width: size.width,
					height: size.height,
					depth: 1,
				}
			}
			Self::T3d(size) => {
				VkExtent3D {
					width: size.width,
					height: size.height,
					depth: size.depth,
				}
			}
			Self::Cube(size) => {
				VkExtent3D {
					width: *size,
					height: *size,
					depth: 1,
				}
			}
			Self::DepthStencil(size) => {
				VkExtent3D {
					width: size.width,
					height: size.height,
					depth: 1,
				}
			}
		}
	}
}

unsafe impl Send for VulkanTextureType {}

/// The wrapper for the Vulkan texture images
pub struct VulkanTexture {
	/// The device holds all of the resource
	pub device: Arc<VulkanDevice>,

	/// The texture image
	pub(crate) image: VkImage,

	/// The image view
	pub(crate) image_view: VkImageView,

	/// The type and size of the texture
	pub(crate) type_size: VulkanTextureType,

	/// The format of the texture
	pub(crate) format: VkFormat,

	/// The memory holds the image data
	pub(crate) memory: Option<VulkanMemory>,
}

impl VulkanTexture {
	/// Create the `VulkanTexture`
	pub fn new(device: Arc<VulkanDevice>, type_size: VulkanTextureType, with_mipmap: bool, format: VkFormat, usage: VkImageUsageFlags) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vkdevice = device.get_vk_device();
		let extent = type_size.get_extent();
		let dim = type_size.get_image_type();
		let is_cube = type_size.is_cube();
		let mipmap_levels = if with_mipmap {
			let mut levels = 0u32;
			let mut size = max(max(extent.width, extent.height), extent.depth);
			while size > 0 {
				size >>= 1;
				levels += 1;
			}
			levels
		} else {
			1
		};
		let image_ci = VkImageCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			pNext: null(),
			flags: if is_cube {VkImageCreateFlagBits::VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT as VkImageCreateFlags} else {0},
			imageType: dim,
			format,
			extent,
			mipLevels: mipmap_levels,
			arrayLayers: 1,
			samples: VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT,
			tiling: VkImageTiling::VK_IMAGE_TILING_OPTIMAL,
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
		let mut ret = Self::new_from_image(device, *image, type_size, format)?;
		ret.memory = Some(memory);
		image.release();
		Ok(ret)
	}

	/// Create the `VulkanTexture` from a image that's not owned (e.g. from a swapchain image)
	pub(crate) fn new_from_image(device: Arc<VulkanDevice>, image: VkImage, type_size: VulkanTextureType, format: VkFormat) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vkdevice = device.get_vk_device();
		let image_view_ci = VkImageViewCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			pNext: null(),
			flags: 0,
			image,
			viewType: if type_size.is_cube() {
				VkImageViewType::VK_IMAGE_VIEW_TYPE_CUBE
			} else {
				match type_size.get_image_type() {
					VkImageType::VK_IMAGE_TYPE_1D => VkImageViewType::VK_IMAGE_VIEW_TYPE_1D,
					VkImageType::VK_IMAGE_TYPE_2D => VkImageViewType::VK_IMAGE_VIEW_TYPE_2D,
					VkImageType::VK_IMAGE_TYPE_3D => VkImageViewType::VK_IMAGE_VIEW_TYPE_3D,
					_ => panic!("Bad image type"),
				}
			},
			format,
			components: VkComponentMapping {
				r: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
				g: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
				b: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
				a: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
			},
			subresourceRange: VkImageSubresourceRange {
				aspectMask: if type_size.is_depth_stencil() {
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as VkImageAspectFlags |
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as VkImageAspectFlags
				} else {
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags
				},
				baseMipLevel: 0,
				levelCount: 1,
				baseArrayLayer: 0,
				layerCount: 1,
			},
		};
		let mut image_view: VkImageView = null();
		vkcore.vkCreateImageView(vkdevice, &image_view_ci, null(), &mut image_view)?;
		Ok(Self {
			device,
			image,
			image_view,
			type_size,
			format,
			memory: None,
		})
	}

	/// Update new data to the texture
	pub fn set_data(&mut self, cmdbuf: VkCommandBuffer, data: *const c_void, offset: &VkOffset3D, extent: &VkExtent3D, size: u64) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let staging_buffer = VulkanBuffer::new(self.device.clone(), size, VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT as u32)?;
		let staging_memory = VulkanMemory::new(self.device.clone(), &staging_buffer.get_memory_requirements()?,
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT as u32 |
			VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT as u32)?;
		staging_memory.bind_vk_buffer(staging_buffer.get_vk_buffer())?;
		staging_memory.set_data(data)?;
		let copy_region = VkBufferImageCopy {
			bufferOffset: 0,
			bufferRowLength: 0,
			bufferImageHeight: 0,
			imageSubresource: VkImageSubresourceLayers {
				aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as u32,
				mipLevel: 0,
				baseArrayLayer: 0,
				layerCount: 1,
			},
			imageOffset: *offset,
			imageExtent: *extent,
		};

		vkcore.vkCmdCopyBufferToImage(cmdbuf, staging_buffer.get_vk_buffer(), self.image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region)?;
		Ok(())
	}

	/// Get the `VkImage`
	pub(crate) fn get_vk_image(&self) -> VkImage {
		self.image
	}

	/// Get the `VkImageView`
	pub(crate) fn get_vk_image_view(&self) -> VkImageView {
		self.image_view
	}

	/// Get if the image is cubemap
	pub fn is_cube(&self) -> bool {
		self.type_size.is_cube()
	}

	/// Get if the image is depth stencil
	pub fn is_depth_stencil(&self) -> bool {
		self.type_size.is_depth_stencil()
	}

	/// Get the `VkImageType`
	pub fn get_image_type(&self) -> VkImageType {
		self.type_size.get_image_type()
	}

	/// Get the extent of the image
	pub fn get_extent(&self) -> VkExtent3D {
		self.type_size.get_extent()
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
