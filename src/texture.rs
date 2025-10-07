
use crate::prelude::*;
use std::{
	any::TypeId,
	cmp::max,
	ffi::{c_void, OsStr},
	fmt::{self, Debug, Formatter},
	fs::read,
	io::Cursor,
	mem::{MaybeUninit, size_of},
	ops::Deref,
	path::{Path, PathBuf},
	ptr::null,
	sync::{
		Arc,
		RwLock,
		atomic::{AtomicBool, Ordering},
	},
};
use image::{
	ImageBuffer,
	Pixel,
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

	/// The image format properties
	pub image_format_props: Option<VkImageFormatProperties>,

	/// The type and size of the texture
	pub(crate) type_size: VulkanTextureType,

	/// The format of the texture
	pub(crate) format: VkFormat,

	/// The memory holds the image data
	pub(crate) memory: Option<VulkanMemory>,

	/// The staging buffer for the texture
	pub staging_buffer: RwLock<Option<StagingBuffer>>,

	/// The mipmap levels
	mipmap_levels: u32,

	/// The mipmap filter
	mipmap_filter: VkFilter,

	/// Is this texture ready to sample by shaders?
	ready_to_sample: AtomicBool,
}

impl VulkanTexture {
	/// Create the `VulkanTexture`
	pub fn new(device: Arc<VulkanDevice>, type_size: VulkanTextureType, with_mipmap: Option<VkFilter>, format: VkFormat, usage: VkImageUsageFlags) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vkdevice = device.get_vk_device();
		let vkphysicaldevice = device.get_vk_physical_device();
		let extent = type_size.get_extent();
		let dim = type_size.get_image_type();
		let is_cube = type_size.is_cube();
		let mipmap_levels = if with_mipmap.is_some() {
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
			usage: usage | VkImageUsageFlagBits::combine(&[
				VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
				VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT
			]),
			sharingMode: VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
			queueFamilyIndexCount: 0,
			pQueueFamilyIndices: null(),
			initialLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
		};
		let mut image_format_props: VkImageFormatProperties = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetPhysicalDeviceImageFormatProperties(vkphysicaldevice, format, dim, image_ci.tiling, image_ci.usage, image_ci.flags, &mut image_format_props)?;
		let mut image: VkImage = null();
		vkcore.vkCreateImage(vkdevice, &image_ci, null(), &mut image)?;
		let image = ResourceGuard::new(image, |&i|vkcore.clone().vkDestroyImage(vkdevice, i, null()).unwrap());
		let mut mem_reqs: VkMemoryRequirements = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetImageMemoryRequirements(vkdevice, *image, &mut mem_reqs)?;
		let memory = VulkanMemory::new(device.clone(), &mem_reqs, VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as u32)?;
		memory.bind_vk_image(*image)?;
		let mut ret = Self::new_from_existing_image(device, *image, type_size, format)?;
		ret.memory = Some(memory);
		ret.mipmap_levels = mipmap_levels;
		ret.mipmap_filter = with_mipmap.unwrap_or(VkFilter::VK_FILTER_LINEAR);
		image.release();
		ret.image_format_props = Some(image_format_props);
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
			image_format_props: None,
			type_size,
			format,
			memory: None,
			staging_buffer: RwLock::new(None),
			mipmap_levels: 1,
			mipmap_filter: VkFilter::VK_FILTER_LINEAR,
			ready_to_sample: AtomicBool::new(false),
		})
	}

	/// Create a texture from image right away
	pub fn new_from_image<P, Container>(device: Arc<VulkanDevice>, cmdbuf: VkCommandBuffer, image: &ImageBuffer<P, Container>, channel_is_normalized: bool, with_mipmap: Option<VkFilter>, usage: VkImageUsageFlags) -> Result<Self, VulkanError>
	where
		P: Pixel,
		Container: Deref<Target = [P::Subpixel]>,
		<P as Pixel>::Subpixel: 'static {
		let (width, height) = image.dimensions();
		let extent = VkExtent2D {
			width,
			height,
		};
		let is_signed;
		let is_float;
		let bits;
		if TypeId::of::<P::Subpixel>() == TypeId::of::<i8>() {
			is_signed = true;
			is_float = false;
			bits = 8;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<u8>() {
			is_signed = false;
			is_float = false;
			bits = 8;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<i16>() {
			is_signed = true;
			is_float = false;
			bits = 16;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<u16>() {
			is_signed = false;
			is_float = false;
			bits = 16;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<i32>() {
			is_signed = true;
			is_float = false;
			bits = 32;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<u32>() {
			is_signed = false;
			is_float = false;
			bits = 32;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<f32>() {
			is_signed = true;
			is_float = true;
			bits = 32;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<i64>() {
			is_signed = true;
			is_float = false;
			bits = 64;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<u64>() {
			is_signed = false;
			is_float = false;
			bits = 64;
		} else if TypeId::of::<P::Subpixel>() == TypeId::of::<f64>() {
			is_signed = true;
			is_float = true;
			bits = 64;
		} else {
			return Err(VulkanError::ImagePixelFormatNotSupported);
		}
		let format = match (P::CHANNEL_COUNT, bits, is_signed, channel_is_normalized, is_float) {
			(1, 8, true,  false, false) => VkFormat::VK_FORMAT_R8_SINT,
			(1, 8, true,  true,  false) => VkFormat::VK_FORMAT_R8_SNORM,
			(1, 8, false, false, false) => VkFormat::VK_FORMAT_R8_UINT,
			(1, 8, false, true,  false) => VkFormat::VK_FORMAT_R8_UNORM,
			(2, 8, true,  false, false) => VkFormat::VK_FORMAT_R8G8_SINT,
			(2, 8, true,  true,  false) => VkFormat::VK_FORMAT_R8G8_SNORM,
			(2, 8, false, false, false) => VkFormat::VK_FORMAT_R8G8_UINT,
			(2, 8, false, true,  false) => VkFormat::VK_FORMAT_R8G8_UNORM,
			(3, 8, true,  false, false) => VkFormat::VK_FORMAT_R8G8B8_SINT,
			(3, 8, true,  true,  false) => VkFormat::VK_FORMAT_R8G8B8_SNORM,
			(3, 8, false, false, false) => VkFormat::VK_FORMAT_R8G8B8_UINT,
			(3, 8, false, true,  false) => VkFormat::VK_FORMAT_R8G8B8_UNORM,
			(4, 8, true,  false, false) => VkFormat::VK_FORMAT_R8G8B8A8_SINT,
			(4, 8, true,  true,  false) => VkFormat::VK_FORMAT_R8G8B8A8_SNORM,
			(4, 8, false, false, false) => VkFormat::VK_FORMAT_R8G8B8A8_UINT,
			(4, 8, false, true,  false) => VkFormat::VK_FORMAT_R8G8B8A8_UNORM,
			(1, 16, true,  false, false) => VkFormat::VK_FORMAT_R16_SINT,
			(1, 16, true,  true,  false) => VkFormat::VK_FORMAT_R16_SNORM,
			(1, 16, false, false, false) => VkFormat::VK_FORMAT_R16_UINT,
			(1, 16, false, true,  false) => VkFormat::VK_FORMAT_R16_UNORM,
			(2, 16, true,  false, false) => VkFormat::VK_FORMAT_R16G16_SINT,
			(2, 16, true,  true,  false) => VkFormat::VK_FORMAT_R16G16_SNORM,
			(2, 16, false, false, false) => VkFormat::VK_FORMAT_R16G16_UINT,
			(2, 16, false, true,  false) => VkFormat::VK_FORMAT_R16G16_UNORM,
			(3, 16, true,  false, false) => VkFormat::VK_FORMAT_R16G16B16_SINT,
			(3, 16, true,  true,  false) => VkFormat::VK_FORMAT_R16G16B16_SNORM,
			(3, 16, false, false, false) => VkFormat::VK_FORMAT_R16G16B16_UINT,
			(3, 16, false, true,  false) => VkFormat::VK_FORMAT_R16G16B16_UNORM,
			(4, 16, true,  false, false) => VkFormat::VK_FORMAT_R16G16B16A16_SINT,
			(4, 16, true,  true,  false) => VkFormat::VK_FORMAT_R16G16B16A16_SNORM,
			(4, 16, false, false, false) => VkFormat::VK_FORMAT_R16G16B16A16_UINT,
			(4, 16, false, true,  false) => VkFormat::VK_FORMAT_R16G16B16A16_UNORM,
			(1, 32, true,  false, false) => VkFormat::VK_FORMAT_R32_SINT,
			(1, 32, false, false, false) => VkFormat::VK_FORMAT_R32_UINT,
			(2, 32, true,  false, false) => VkFormat::VK_FORMAT_R32G32_SINT,
			(2, 32, false, false, false) => VkFormat::VK_FORMAT_R32G32_UINT,
			(3, 32, true,  false, false) => VkFormat::VK_FORMAT_R32G32B32_SINT,
			(3, 32, false, false, false) => VkFormat::VK_FORMAT_R32G32B32_UINT,
			(4, 32, true,  false, false) => VkFormat::VK_FORMAT_R32G32B32A32_SINT,
			(4, 32, false, false, false) => VkFormat::VK_FORMAT_R32G32B32A32_UINT,
			(1, 64, true,  false, false) => VkFormat::VK_FORMAT_R64_SINT,
			(1, 64, false, false, false) => VkFormat::VK_FORMAT_R64_UINT,
			(2, 64, true,  false, false) => VkFormat::VK_FORMAT_R64G64_SINT,
			(2, 64, false, false, false) => VkFormat::VK_FORMAT_R64G64_UINT,
			(3, 64, true,  false, false) => VkFormat::VK_FORMAT_R64G64B64_SINT,
			(3, 64, false, false, false) => VkFormat::VK_FORMAT_R64G64B64_UINT,
			(4, 64, true,  false, false) => VkFormat::VK_FORMAT_R64G64B64A64_SINT,
			(4, 64, false, false, false) => VkFormat::VK_FORMAT_R64G64B64A64_UINT,
			(1, 32, true, _, true) => VkFormat::VK_FORMAT_R32_SFLOAT,
			(2, 32, true, _, true) => VkFormat::VK_FORMAT_R32G32_SFLOAT,
			(3, 32, true, _, true) => VkFormat::VK_FORMAT_R32G32B32_SFLOAT,
			(4, 32, true, _, true) => VkFormat::VK_FORMAT_R32G32B32A32_SFLOAT,
			(1, 64, true, _, true) => VkFormat::VK_FORMAT_R64_SFLOAT,
			(2, 64, true, _, true) => VkFormat::VK_FORMAT_R64G64_SFLOAT,
			(3, 64, true, _, true) => VkFormat::VK_FORMAT_R64G64B64_SFLOAT,
			(4, 64, true, _, true) => VkFormat::VK_FORMAT_R64G64B64A64_SFLOAT,
			_ => return Err(VulkanError::ImagePixelFormatNotSupported),
		};
		let ret = Self::new(device, VulkanTextureType::T2d(extent), with_mipmap, format, usage)?;
		ret.set_staging_data(image.as_ptr() as *const c_void, 0, (bits / 8 * P::CHANNEL_COUNT) as usize * width as usize)?;
		let offset = VkOffset3D {
			x: 0,
			y: 0,
			z: 0,
		};
		let update_extent = VkExtent3D {
			width,
			height,
			depth: 1,
		};
		ret.upload_staging_buffer(cmdbuf, &offset, &update_extent)?;
		Ok(ret)
	}

	/// Create a texture from image loaded from file path right away
	pub fn new_from_path<P: AsRef<Path>>(device: Arc<VulkanDevice>, cmdbuf: VkCommandBuffer, path: P, channel_is_normalized: bool, with_mipmap: Option<VkFilter>, usage: VkImageUsageFlags) -> Result<Self, VulkanError> {
		let image_data = read(&path)?;
		let pb = PathBuf::from(path.as_ref());
		let image = if pb.extension().and_then(OsStr::to_str).map(|s| {let s = s.to_lowercase(); s != "jpg" && s != "jpeg"}).unwrap_or(true) {
			use image::DynamicImage;
			let img = image::ImageReader::new(Cursor::new(&image_data)).with_guessed_format()?.decode()?;
			match img {
				DynamicImage::ImageLuma8(img) => Self::new_from_image(device, cmdbuf, &img, channel_is_normalized, with_mipmap, usage),
				DynamicImage::ImageLumaA8(img) => Self::new_from_image(device, cmdbuf, &img, channel_is_normalized, with_mipmap, usage),
				DynamicImage::ImageLuma16(img) => Self::new_from_image(device, cmdbuf, &img, channel_is_normalized, with_mipmap, usage),
				DynamicImage::ImageLumaA16(img) => Self::new_from_image(device, cmdbuf, &img, channel_is_normalized, with_mipmap, usage),
				DynamicImage::ImageRgba8(img) => Self::new_from_image(device, cmdbuf, &img, channel_is_normalized, with_mipmap, usage),
				DynamicImage::ImageRgba16(img) => Self::new_from_image(device, cmdbuf, &img, channel_is_normalized, with_mipmap, usage),
				DynamicImage::ImageRgba32F(img) => Self::new_from_image(device, cmdbuf, &img, channel_is_normalized, with_mipmap, usage),
				DynamicImage::ImageRgb8(img) => {
					use image::{Rgba, RgbaImage};
					let rgba_img: RgbaImage = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
						let pixel = img.get_pixel(x, y);
						Rgba([pixel[0], pixel[1], pixel[2], 255])
					});
					Self::new_from_image(device, cmdbuf, &rgba_img, channel_is_normalized, with_mipmap, usage)
				}
				DynamicImage::ImageRgb16(img) => {
					use image::Rgba;
					type Rgba16Image = ImageBuffer<Rgba<u16>, Vec<u16>>;
					let rgba_img: Rgba16Image = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
						let pixel = img.get_pixel(x, y);
						Rgba([pixel[0], pixel[1], pixel[2], 65535])
					});
					Self::new_from_image(device, cmdbuf, &rgba_img, channel_is_normalized, with_mipmap, usage)
				}
				DynamicImage::ImageRgb32F(img) => {
					use image::{Rgba, Rgba32FImage};
					let rgba_img: Rgba32FImage = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
						let pixel = img.get_pixel(x, y);
						Rgba([pixel[0], pixel[1], pixel[2], 1.0])
					});
					Self::new_from_image(device, cmdbuf, &rgba_img, channel_is_normalized, with_mipmap, usage)
				}
				_ => Err(VulkanError::LoadImageFailed(format!("Unknown image type: {img:?}")))
			}
		} else {
			use image::{Rgba, RgbaImage};
			let img: image::RgbImage = turbojpeg::decompress_image(&image_data).unwrap();
			let rgba_img: RgbaImage = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
				let pixel = img.get_pixel(x, y);
				Rgba([pixel[0], pixel[1], pixel[2], 255])
			});
			Self::new_from_image(device, cmdbuf, &rgba_img, channel_is_normalized, with_mipmap, usage)
		}?;
		Ok(image)
	}

	/// Get the size of the image
	pub fn get_size(&self) -> Result<VkDeviceSize, VulkanError> {
		let mut mem_reqs: VkMemoryRequirements = unsafe {MaybeUninit::zeroed().assume_init()};
		self.device.vkcore.vkGetImageMemoryRequirements(self.device.get_vk_device(), self.image, &mut mem_reqs)?;
		Ok(mem_reqs.size)
	}

	/// Get the mipmap levels
	pub fn get_mipmap_levels(&self) -> u32 {
		self.mipmap_levels
	}

	/// Create the staging buffer if it not exists
	pub fn ensure_staging_buffer(&self) -> Result<(), VulkanError> {
		let mut staging_buffer = self.staging_buffer.write().unwrap();
		if staging_buffer.is_none() {
			*staging_buffer = Some(StagingBuffer::new(self.device.clone(), self.get_size()?)?);
		}
		Ok(())
	}

	/// Get the data pointer of the staging buffer
	pub fn get_staging_buffer_address(&self) -> Result<*mut c_void, VulkanError> {
		self.ensure_staging_buffer()?;
		Ok(self.staging_buffer.read().unwrap().as_ref().unwrap().address)
	}

	/// Discard the staging buffer to save memory
	pub fn discard_staging_buffer(&self) {
		let mut staging_buffer = self.staging_buffer.write().unwrap();
		*staging_buffer = None;
	}

	/// Update new data to the staging buffer
	pub fn set_staging_data(&self, data: *const c_void, offset: VkDeviceSize, size: usize) -> Result<(), VulkanError> {
		self.ensure_staging_buffer()?;
		self.staging_buffer.write().unwrap().as_mut().unwrap().set_data(data, offset, size)?;
		Ok(())
	}

	/// Update new data to the staging buffer from a `RgbImage`
	pub fn set_staging_data_from_image<P, Container>(&self, image: &ImageBuffer<P, Container>, z_layer: u32) -> Result<(), VulkanError>
	where
		P: Pixel,
		Container: Deref<Target = [P::Subpixel]> {
		let (width, height) = image.dimensions();
		let extent = self.type_size.get_extent();
		let row_byte_count = size_of::<P>() * width as usize;
		let image_size = row_byte_count * height as usize;
		if width != extent.width || height != extent.height {
			return Err(VulkanError::ImageTypeSizeNotMatch(format!("The size of the texture is {extent:?}, but the size of the image is {width}x_{height}. The size doesn't match.")));
		}
		if z_layer >= extent.depth {
			panic!("The given `z_layer` is {z_layer}, but the depth of the texture is {}, the `z_layer` is out of bound", extent.depth);
		}
		let layer_offset = z_layer as usize * image_size;
		let image_address = image.as_ptr() as *const c_void;
		self.set_staging_data(image_address, layer_offset as u64, image_size)
	}

	/// Upload the staging buffer data to the texture
	pub fn upload_staging_buffer(&self, cmdbuf: VkCommandBuffer, offset: &VkOffset3D, extent: &VkExtent3D) -> Result<(), VulkanError> {
		let staging_buffer_lock = self.staging_buffer.read().unwrap();
		if let Some(ref staging_buffer) = *staging_buffer_lock {
			let barrier = VkImageMemoryBarrier {
				sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				pNext: null(),
				srcAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT as VkAccessFlags,
				dstAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags,
				oldLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
				newLayout: VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				image: self.image,
				subresourceRange: VkImageSubresourceRange {
					aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
					baseMipLevel: 0,
					levelCount: 1,
					baseArrayLayer: 0,
					layerCount: 1,
				},
			};

			self.device.vkcore.vkCmdPipelineBarrier(cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				0,
				0, null(),
				0, null(),
				1, &barrier
			)?;

			self.ready_to_sample.store(false, Ordering::Relaxed);

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

			self.device.vkcore.vkCmdCopyBufferToImage(cmdbuf, staging_buffer.get_vk_buffer(), self.image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region)?;
			if self.mipmap_levels > 1 {
				self.generate_mipmaps(cmdbuf)?;
			}
			Ok(())
		} else {
			Err(VulkanError::NoStagingBuffer)
		}
	}

	/// Upload the staging buffer data to the texture
	pub fn upload_staging_buffer_multi(&self, cmdbuf: VkCommandBuffer, regions: &[TextureRegion]) -> Result<(), VulkanError> {
		let staging_buffer_lock = self.staging_buffer.read().unwrap();
		if let Some(ref staging_buffer) = *staging_buffer_lock {
			let barrier = VkImageMemoryBarrier {
				sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				pNext: null(),
				srcAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT as VkAccessFlags,
				dstAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags,
				oldLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
				newLayout: VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				image: self.image,
				subresourceRange: VkImageSubresourceRange {
					aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
					baseMipLevel: 0,
					levelCount: 1,
					baseArrayLayer: 0,
					layerCount: 1,
				},
			};

			self.device.vkcore.vkCmdPipelineBarrier(cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				0,
				0, null(),
				0, null(),
				1, &barrier
			)?;

			self.ready_to_sample.store(false, Ordering::Relaxed);

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

			self.device.vkcore.vkCmdCopyBufferToImage(cmdbuf, staging_buffer.get_vk_buffer(), self.image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy_regions.len() as u32, copy_regions.as_ptr())?;
			if self.mipmap_levels > 1 {
				self.generate_mipmaps(cmdbuf)?;
			}
			Ok(())
		} else {
			Err(VulkanError::NoStagingBuffer)
		}
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

	/// Generate mipmaps for the texture
	pub fn generate_mipmaps(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		let extent = self.type_size.get_extent();

		let mut mip_width = extent.width;
		let mut mip_height = extent.height;
		let mut mip_depth = extent.depth;

		for i in 1..self.mipmap_levels {
			let barrier_base = VkImageMemoryBarrier {
				sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				pNext: null(),
				srcAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags,
				dstAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT as VkAccessFlags,
				oldLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
				newLayout: VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				image: self.image,
				subresourceRange: VkImageSubresourceRange {
					aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
					baseMipLevel: i - 1,
					levelCount: 1,
					baseArrayLayer: 0,
					layerCount: 1,
				},
			};

			let barrier_curr = VkImageMemoryBarrier {
				sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				pNext: null(),
				srcAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT as VkAccessFlags,
				dstAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags,
				oldLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
				newLayout: VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				image: self.image,
				subresourceRange: VkImageSubresourceRange {
					aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
					baseMipLevel: i,
					levelCount: 1,
					baseArrayLayer: 0,
					layerCount: 1,
				},
			};

			self.device.vkcore.vkCmdPipelineBarrier(cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				0,
				0, null(),
				0, null(),
				1, &barrier_base
			)?;

			self.device.vkcore.vkCmdPipelineBarrier(cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				0,
				0, null(),
				0, null(),
				1, &barrier_curr
			)?;

			let next_mip_width = max(mip_width / 2, 1);
			let next_mip_height = max(mip_height / 2, 1);
			let next_mip_depth = max(mip_depth / 2, 1);

			let blit = VkImageBlit {
				srcSubresource: VkImageSubresourceLayers {
					aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
					mipLevel: i - 1,
					baseArrayLayer: 0,
					layerCount: 1,
				},
				srcOffsets: [
					VkOffset3D {
						x: 0,
						y: 0,
						z: 0,
					},
					VkOffset3D {
						x: mip_width as i32,
						y: mip_height as i32,
						z: mip_depth as i32,
					},
				],
				dstSubresource: VkImageSubresourceLayers {
					aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
					mipLevel: i,
					baseArrayLayer: 0,
					layerCount: 1,
				},
				dstOffsets: [
					VkOffset3D {
						x: 0,
						y: 0,
						z: 0,
					},
					VkOffset3D {
						x: next_mip_width as i32,
						y: next_mip_height as i32,
						z: next_mip_depth as i32,
					},
				],
			};

			self.device.vkcore.vkCmdBlitImage(cmdbuf,
				self.image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				self.image, VkImageLayout::VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				self.mipmap_filter
			)?;

			mip_width = next_mip_width;
			mip_height = next_mip_height;
			mip_depth = next_mip_depth;
		}

		self.ready_to_sample.store(false, Ordering::Relaxed);
		Ok(())
	}

	/// Make this texture ready to be sampled by shaders
	pub fn prepare_for_sample(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		if self.ready_to_sample.load(Ordering::Relaxed) == false {
			let barrier = VkImageMemoryBarrier {
				sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				pNext: null(),
				srcAccessMask: VkAccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT as VkAccessFlags,
				dstAccessMask: VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT as VkAccessFlags,
				oldLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
				newLayout: VkImageLayout::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
				image: self.image,
				subresourceRange: VkImageSubresourceRange {
					aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as VkImageAspectFlags,
					baseMipLevel: 0,
					levelCount: self.mipmap_levels,
					baseArrayLayer: 0,
					layerCount: 1,
				},
			};

			self.device.vkcore.vkCmdPipelineBarrier(cmdbuf,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT as VkPipelineStageFlags,
				VkPipelineStageFlagBits::VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT as VkPipelineStageFlags,
				0,
				0, null(),
				0, null(),
				1, &barrier
			)?;
			self.ready_to_sample.store(true, Ordering::Relaxed);
		}
		Ok(())
	}
}

unsafe impl Send for VulkanTexture {}
unsafe impl Sync for VulkanTexture {}

impl Debug for VulkanTexture {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanTexture")
		.field("image", &self.image)
		.field("image_view", &self.image_view)
		.field("image_format_props", &self.image_format_props)
		.field("type_size", &self.type_size)
		.field("format", &self.format)
		.field("memory", &self.memory)
		.field("staging_buffer", &self.staging_buffer)
		.field("mipmap_levels", &self.mipmap_levels)
		.field("mipmap_filter", &self.mipmap_filter)
		.field("ready_to_sample", &self.ready_to_sample)
		.finish()
	}
}

impl Drop for VulkanTexture {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		let vkdevice = self.device.get_vk_device();
		proceed_run(vkcore.vkDestroyImageView(vkdevice, self.image_view, null()));

		// Only destroy the image if it was owned by the struct.
		if self.memory.is_some() {
			proceed_run(vkcore.vkDestroyImage(vkdevice, self.image, null()));
		}
	}
}
