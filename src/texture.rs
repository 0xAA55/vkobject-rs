
use crate::prelude::*;
use std::{
	cmp::max,
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::MaybeUninit,
	ptr::null,
	sync::Arc,
};

/// The offset and extent of a piece of the texture
#[derive(Debug, Clone, Copy)]
pub struct TextureRegion {
	pub offset: VkOffset3D,
	pub extent: VkExtent3D,
}

/// The texture type and size
#[derive(Debug, Clone, Copy)]
pub enum VulkanTextureType {
	/// 1D texture
	T1d(u32),

	/// 2D texture
	T2d(VkExtent2D),

	/// 3D texture (a.k.a. volume texture)
	T3d(VkExtent3D),

	/// Cubemap texture with 6 faces
	Cube(u32),

	/// The texture dedicated for a depth stencil buffer
	DepthStencil(VkExtent2D),
}

impl VulkanTextureType {
	/// Get if the image is cubemap
	pub fn is_cube(&self) -> bool {
		matches!(self, Self::Cube(_))
	}

	/// Get if the image is depth stencil
	pub fn is_depth_stencil(&self) -> bool {
		matches!(self, Self::DepthStencil(_))
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

	/// The staging buffer for the texture
	pub staging_buffer: Option<StagingBuffer>,
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
		let mut ret = Self::new_from_existing_image(device, *image, type_size, format)?;
		ret.memory = Some(memory);
		image.release();
		Ok(ret)
	}

	/// Create the `VulkanTexture` from a image that's not owned (e.g. from a swapchain image)
	pub(crate) fn new_from_existing_image(device: Arc<VulkanDevice>, image: VkImage, type_size: VulkanTextureType, format: VkFormat) -> Result<Self, VulkanError> {
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
			staging_buffer: None,
		})
	}

	/// Get the size of the image
	pub fn get_size(&self) -> Result<VkDeviceSize, VulkanError> {
		let mut mem_reqs: VkMemoryRequirements = unsafe {MaybeUninit::zeroed().assume_init()};
		self.device.vkcore.vkGetImageMemoryRequirements(self.device.get_vk_device(), self.image, &mut mem_reqs)?;
		Ok(mem_reqs.size)
	}

	/// Get the pitch, the bytes per row of the texture
	pub fn get_pitch(&self) -> Result<usize, VulkanError> {
		let extent = self.type_size.get_extent();
		Ok(self.get_size()? as usize / extent.depth as usize / extent.height as usize)
	}

	/// Create the staging buffer if it not exists
	pub fn ensure_staging_buffer(&mut self) -> Result<(), VulkanError> {
		if self.staging_buffer.is_none() {
			self.staging_buffer = Some(StagingBuffer::new(self.device.clone(), self.get_size()?)?);
		}
		Ok(())
	}

	/// Get the data pointer of the staging buffer
	pub fn get_staging_buffer_address(&mut self) -> Result<*mut c_void, VulkanError> {
		self.ensure_staging_buffer()?;
		Ok(self.staging_buffer.as_ref().unwrap().address)
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&mut self) {
		self.staging_buffer = None;
	}

	/// Update new data to the staging buffer
	pub fn set_staging_data(&mut self, data: *const c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		self.ensure_staging_buffer()?;
		self.staging_buffer.as_ref().unwrap().set_data(data, offset, size)?;
		Ok(())
	}

	/// Upload the staging buffer data to the texture
	pub fn upload_staging_buffer(&self, cmdbuf: VkCommandBuffer, offset: &VkOffset3D, extent: &VkExtent3D) -> Result<(), VulkanError> {
		let copy_region = VkBufferImageCopy {
			bufferOffset: 0,
			bufferRowLength: 0,
			bufferImageHeight: 0,
			imageSubresource: VkImageSubresourceLayers {
				aspectMask: if self.is_depth_stencil() {
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as VkImageAspectFlags |
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as VkImageAspectFlags
				} else {
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags
				},
				mipLevel: 0,
				baseArrayLayer: 0,
				layerCount: 1,
			},
			imageOffset: *offset,
			imageExtent: *extent,
		};

		self.device.vkcore.vkCmdCopyBufferToImage(cmdbuf, self.staging_buffer.as_ref().unwrap().get_vk_buffer(), self.image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region)?;
		Ok(())
	}

	/// Upload the staging buffer data to the texture
	pub fn upload_staging_buffer_multi(&self, cmdbuf: VkCommandBuffer, regions: &[TextureRegion]) -> Result<(), VulkanError> {
		let copy_regions: Vec<VkBufferImageCopy> = regions.iter().map(|r| VkBufferImageCopy {
			bufferOffset: 0,
			bufferRowLength: 0,
			bufferImageHeight: 0,
			imageSubresource: VkImageSubresourceLayers {
				aspectMask: if self.is_depth_stencil() {
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as VkImageAspectFlags |
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as VkImageAspectFlags
				} else {
					VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags
				},
				mipLevel: 0,
				baseArrayLayer: 0,
				layerCount: 1,
			},
			imageOffset: r.offset,
			imageExtent: r.extent,
		}).collect();

		self.device.vkcore.vkCmdCopyBufferToImage(cmdbuf, self.staging_buffer.as_ref().unwrap().get_vk_buffer(), self.image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy_regions.len() as u32, copy_regions.as_ptr())?;
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

	/// Get the type and size of the texture
	pub fn get_type_size(&self) -> VulkanTextureType {
		self.type_size
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

unsafe impl Send for VulkanTexture {}
unsafe impl Sync for VulkanTexture {}

impl Debug for VulkanTexture {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanTexture")
		.field("image", &self.image)
		.field("image_view", &self.image_view)
		.field("type_size", &self.type_size)
		.field("format", &self.format)
		.field("memory", &self.memory)
		.field("staging_buffer", &self.staging_buffer)
		.finish()
	}
}

impl Drop for VulkanTexture {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		let vkdevice = self.device.get_vk_device();
		vkcore.vkDestroyImageView(vkdevice, self.image_view, null()).unwrap();

		// Only destroy the image if it was owned by the struct.
		if self.memory.is_some() {
			vkcore.vkDestroyImage(vkdevice, self.image, null()).unwrap();
		}
	}
}
