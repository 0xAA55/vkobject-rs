
use crate::prelude::*;
use std::{
	cmp::max,
	fmt::{self, Debug, Formatter},
	mem::{MaybeUninit, transmute, swap},
	ptr::{null, null_mut},
	sync::{Arc, Mutex},
};

/// An image of a swapchain that's dedicated for the depth-stencil usage
pub struct VulkanDepthStencilImage {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the image
	image: VkImage,

	/// The handle to the image view
	image_view: VkImageView,

	/// The video memory of the image
	memory: VulkanMemory,
}

impl VulkanDepthStencilImage {
	pub fn new(device: Arc<VulkanDevice>, extent: &VkExtent2D, format: VkFormat) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vkdevice = device.get_vk_device();
		let image_ci = VkImageCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			imageType: VkImageType::VK_IMAGE_TYPE_2D,
			format,
			extent: VkExtent3D {
				width: extent.width,
				height: extent.height,
				depth: 1,
			},
			mipLevels: 1,
			arrayLayers: 1,
			samples: VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT,
			tiling: VkImageTiling::VK_IMAGE_TILING_OPTIMAL,
			usage: VkImageUsageFlagBits::VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT as u32,
			sharingMode: VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
			queueFamilyIndexCount: 0,
			pQueueFamilyIndices: null(),
			initialLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
		};
		let mut image: VkImage = null();
		vkcore.vkCreateImage(vkdevice, &image_ci, null(), &mut image)?;
		let image = ResourceGuard::new(image, |&i|vkcore.vkDestroyImage(vkdevice, i, null()).unwrap());
		let mut mem_reqs: VkMemoryRequirements = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetImageMemoryRequirements(vkdevice, *image, &mut mem_reqs)?;
		let memory = VulkanMemory::new(device.clone(), &mem_reqs, VkMemoryPropertyFlagBits::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT as u32)?;
		let memory = ResourceGuard::new(memory, |m|vkcore.vkFreeMemory(vkdevice, m.get_vk_memory(), null()).unwrap());
		vkcore.vkBindImageMemory(vkdevice, *image, memory.get_vk_memory(), 0)?;
		let image_view_ci = VkImageViewCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			pNext: null(),
			flags: 0,
			image: *image,
			viewType: VkImageViewType::VK_IMAGE_VIEW_TYPE_2D,
			format,
			components: VkComponentMapping {
				r: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
				g: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
				b: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
				a: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_IDENTITY,
			},
			subresourceRange: VkImageSubresourceRange {
				aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_DEPTH_BIT as u32 | VkImageAspectFlagBits::VK_IMAGE_ASPECT_STENCIL_BIT as u32,
				baseMipLevel: 0,
				levelCount: 1,
				baseArrayLayer: 0,
				layerCount: 1,
			},
		};
		let mut image_view: VkImageView = null();
		vkcore.vkCreateImageView(vkdevice, &image_view_ci, null(), &mut image_view)?;
		let image = image.release();
		let memory = memory.release();
		Ok(Self {
			device,
			image,
			image_view,
			memory,
		})
	}

	/// Get the `VkImage`
	pub(crate) fn get_vk_image(&self) -> VkImage {
		self.image
	}

	/// Get the `VkImageView`
	pub(crate) fn get_vk_image_view(&self) -> VkImageView {
		self.image_view
	}
}

impl Debug for VulkanDepthStencilImage {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanDepthStencilImage")
		.field("image", &self.image)
		.field("image_view", &self.image_view)
		.field("memory", &self.memory)
		.finish()
	}
}

impl Drop for VulkanDepthStencilImage {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		let vkdevice = self.device.get_vk_device();
		vkcore.vkDestroyImageView(vkdevice, self.image_view, null()).unwrap();
		vkcore.vkDestroyImage(vkdevice, self.image, null()).unwrap();
	}
}

unsafe impl Send for VulkanDepthStencilImage {}

/// An image of a swapchain
pub struct VulkanSwapchainImage {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the image
	image: VkImage,

	/// The handle to the image view
	image_view: VkImageView,

	/// The depth stencil image
	depth_stencil: VulkanDepthStencilImage,

	/// The framebuffer
	framebuffer: VulkanFramebuffer,

	/// The renderpass to the framebuffer
	renderpass: VulkanRenderPass,

	/// The extent of the image
	extent: VkExtent2D,

	/// The semaphore to acquire the image for rendering
	pub acquire_semaphore: Arc<VulkanSemaphore>,

	/// The semaphore for the image on release
	pub release_semaphore: Arc<VulkanSemaphore>,

	/// The fence of submitting commands to a queue
	pub queue_submit_fence: Arc<VulkanFence>,
}

unsafe impl Send for VulkanSwapchainImage {}

impl VulkanSwapchainImage {
	/// Create the `VulkanSwapchainImage`
	pub fn new(device: Arc<VulkanDevice>, surface_format: &VkSurfaceFormatKHR, image: VkImage, extent: &VkExtent2D, depth_stencil_format: VkFormat) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let vk_device = device.get_vk_device();
		let vk_image_view_ci = VkImageViewCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			pNext: null(),
			flags: 0,
			image,
			viewType: VkImageViewType::VK_IMAGE_VIEW_TYPE_2D,
			format: surface_format.format,
			components: VkComponentMapping {
				r: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_R,
				g: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_G,
				b: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_B,
				a: VkComponentSwizzle::VK_COMPONENT_SWIZZLE_A,
			},
			subresourceRange: VkImageSubresourceRange {
				aspectMask: VkImageAspectFlagBits::VK_IMAGE_ASPECT_COLOR_BIT as u32,
				baseMipLevel: 0,
				levelCount: 1,
				baseArrayLayer: 0,
				layerCount: 1,
			},
		};
		let mut image_view: VkImageView = null();
		vkcore.vkCreateImageView(vk_device, &vk_image_view_ci, null(), &mut image_view)?;
		let image_view = ResourceGuard::new(image_view, |&i|vkcore.clone().vkDestroyImageView(vk_device, i, null()).unwrap());
		let acquire_semaphore = Arc::new(VulkanSemaphore::new(device.clone())?);
		let release_semaphore = Arc::new(VulkanSemaphore::new(device.clone())?);
		let queue_submit_fence = Arc::new(VulkanFence::new(device.clone())?);
		let depth_stencil = VulkanDepthStencilImage::new(device.clone(), extent, depth_stencil_format)?;
		let renderpass_attachments = [
			VulkanRenderPassAttachment::new(surface_format.format, false),
			VulkanRenderPassAttachment::new(depth_stencil_format, true),
		];
		let renderpass = VulkanRenderPass::new(device.clone(), &renderpass_attachments)?;
		let attachments = [*image_view, depth_stencil.image_view];
		let framebuffer = VulkanFramebuffer::new(device.clone(), extent, renderpass.get_vk_renderpass(), &attachments)?;
		let image_view = image_view.release();
		Ok(Self{
			device,
			image,
			image_view,
			depth_stencil,
			framebuffer,
			renderpass,
			extent: *extent,
			acquire_semaphore,
			release_semaphore,
			queue_submit_fence,
		})
	}

	/// Get the `VkImage`
	pub(crate) fn get_vk_image(&self) -> VkImage {
		self.image
	}

	/// Get the `VkImageView`
	pub(crate) fn get_vk_image_view(&self) -> VkImageView {
		self.image_view
	}

	/// Get the extent of the image
	pub fn get_extent(&self) -> &VkExtent2D {
		&self.extent
	}
}

impl Debug for VulkanSwapchainImage {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanSwapchainImage")
		.field("image", &self.image)
		.field("image_view", &self.image_view)
		.field("depth_stencil", &self.depth_stencil)
		.field("framebuffer", &self.framebuffer)
		.field("renderpass", &self.renderpass)
		.field("extent", &self.extent)
		.field("acquire_semaphore", &self.acquire_semaphore)
		.field("release_semaphore", &self.release_semaphore)
		.field("queue_submit_fence", &self.queue_submit_fence)
		.finish()
	}
}

impl Drop for VulkanSwapchainImage {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		vkcore.vkDestroyImageView(self.device.get_vk_device(), self.image_view, null()).unwrap();
	}
}

/// A swapchain for presenting frames to the window surface.
pub struct VulkanSwapchain {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The `VulkanSurface` that is needed by the swapchain
	surface: Arc<VulkanSurface>,

	/// Is VSYNC on?
	vsync: bool,

	/// Is this swapchain for VR?
	is_vr: bool,

	/// The capabilities of the surface
	surf_caps: VkSurfaceCapabilitiesKHR,

	/// The handle to the swapchain
	swapchain: VkSwapchainKHR,

	/// The extent of the swapchain
	swapchain_extent: VkExtent2D,

	/// The current present mode of the swapchain
	present_mode: VkPresentModeKHR,

	/// The depth stencil format
	depth_stencil_format: VkFormat,

	/// The desired image count
	desired_num_of_swapchain_images: u32,

	/// The swapchain images
	pub images: Vec<Arc<Mutex<VulkanSwapchainImage>>>,

	/// The semaphore for acquiring new frame image
	acquire_semaphore: Arc<VulkanSemaphore>,

	/// The current image index in use
	cur_image_index: u32,
}

unsafe impl Send for VulkanSwapchain {}

impl VulkanSwapchain {
	/// Create the `VulkanSwapchain`
	pub fn new(device: Arc<VulkanDevice>, surface: Arc<VulkanSurface>, width: u32, height: u32, vsync: bool, is_vr: bool, num_images: u32, old_swapchain: Option<VkSwapchainKHR>) -> Result<Self, VulkanError> {
		let vkcore = device.vkcore.clone();
		let surface_format = *surface.get_vk_surface_format();
		let vk_device = device.get_vk_device();
		let vk_phy_dev = device.get_vk_physical_device();
		let vk_surface = surface.get_vk_surface();

		let mut surf_caps: VkSurfaceCapabilitiesKHR  = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk_phy_dev, vk_surface, &mut surf_caps)?;

		let swapchain_extent = if surf_caps.currentExtent.width == u32::MAX {
			VkExtent2D {
				width,
				height,
			}
		} else {
			surf_caps.currentExtent
		};

		let mut num_present_mode = 0u32;
		vkcore.vkGetPhysicalDeviceSurfacePresentModesKHR(vk_phy_dev, vk_surface, &mut num_present_mode, null_mut())?;
		let mut present_modes = Vec::<VkPresentModeKHR>::with_capacity(num_present_mode as usize);
		vkcore.vkGetPhysicalDeviceSurfacePresentModesKHR(vk_phy_dev, vk_surface, &mut num_present_mode, present_modes.as_mut_ptr())?;
		unsafe {present_modes.set_len(num_present_mode as usize)};

		// Select a present mode for the swapchain
		let mut present_mode = VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR;

		// If v-sync is not requested, try to find a mailbox mode
		// It's the lowest latency non-tearing present mode available
		if !vsync {
			for mode in present_modes.iter() {
				if *mode == VkPresentModeKHR::VK_PRESENT_MODE_MAILBOX_KHR || *mode == VkPresentModeKHR::VK_PRESENT_MODE_IMMEDIATE_KHR {
					present_mode = *mode;
					break;
				}
			}
		}

		// Determine the number of images
		let mut desired_num_of_swapchain_images = max(num_images, surf_caps.minImageCount + 1);
		if surf_caps.maxImageCount > 0 && desired_num_of_swapchain_images > surf_caps.maxImageCount {
			desired_num_of_swapchain_images = surf_caps.maxImageCount;
		}

		// Find the transformation of the surface
		let pre_transform: VkSurfaceTransformFlagBitsKHR = unsafe {transmute(if (surf_caps.supportedTransforms &
			VkSurfaceTransformFlagBitsKHR::VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR as u32) ==
			VkSurfaceTransformFlagBitsKHR::VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR as u32 {
			VkSurfaceTransformFlagBitsKHR::VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
		} else {
			surf_caps.currentTransform
		} as u32)};

		// Find a supported composite alpha format (not all devices support alpha opaque)
		let mut composite_alpha = VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		const COMPOSITE_ALPHA_FLAGS: [VkCompositeAlphaFlagBitsKHR; 4] = [
			VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
			VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
			VkCompositeAlphaFlagBitsKHR::VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
		];
		for flag in COMPOSITE_ALPHA_FLAGS.iter() {
			if (surf_caps.supportedCompositeAlpha & *flag as u32) == *flag as u32 {
				composite_alpha = *flag;
				break;
			}
		}

		let swapchain_ci = VkSwapchainCreateInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			pNext: null(),
			flags: 0,
			surface: vk_surface,
			minImageCount: desired_num_of_swapchain_images,
			imageFormat: surface_format.format,
			imageColorSpace: surface_format.colorSpace,
			imageExtent: swapchain_extent,
			imageArrayLayers: if !is_vr {1} else {2},
			imageUsage: VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT as u32 |
				(surf_caps.supportedUsageFlags & VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_SRC_BIT as u32) |
				(surf_caps.supportedUsageFlags & VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT as u32)
				as VkImageUsageFlags,
			imageSharingMode: VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
			queueFamilyIndexCount: 0,
			pQueueFamilyIndices: null(),
			preTransform: pre_transform,
			compositeAlpha: composite_alpha,
			presentMode: present_mode,
			clipped: VK_TRUE,
			oldSwapchain: match old_swapchain {
				Some(chain) => chain,
				None => null(),
			},
		};

		let mut swapchain: VkSwapchainKHR = null();
		vkcore.vkCreateSwapchainKHR(vk_device, &swapchain_ci, null(), &mut swapchain)?;
		let swapchain = ResourceGuard::new(swapchain, |&s|vkcore.vkDestroySwapchainKHR(vk_device, s, null()).unwrap());
		let mut num_images = 0u32;
		vkcore.vkGetSwapchainImagesKHR(vk_device, *swapchain, &mut num_images, null_mut())?;
		let mut vk_images = Vec::<VkImage>::with_capacity(num_images as usize);
		vkcore.vkGetSwapchainImagesKHR(vk_device, *swapchain, &mut num_images, vk_images.as_mut_ptr())?;
		unsafe {vk_images.set_len(num_images as usize)};
		let mut images = Vec::<Arc<Mutex<VulkanSwapchainImage>>>::with_capacity(vk_images.len());
		let depth_stencil_format = Self::get_depth_stencil_format(&vkcore, vk_phy_dev)?;
		for vk_image in vk_images.iter() {
			images.push(Arc::new(Mutex::new(VulkanSwapchainImage::new(device.clone(), &surface_format, *vk_image, &swapchain_extent, depth_stencil_format)?)));
		}
		let acquire_semaphore = Arc::new(VulkanSemaphore::new(device.clone())?);
		let swapchain = swapchain.release();

		Ok(Self {
			device,
			surface,
			vsync,
			is_vr,
			surf_caps,
			swapchain,
			swapchain_extent,
			present_mode,
			depth_stencil_format,
			desired_num_of_swapchain_images,
			images,
			acquire_semaphore,
			cur_image_index: 0,
		})
	}

	/// Get the `VkSwapchainKHR`
	pub(crate) fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.swapchain
	}

	/// Get the `VkSurfaceCapabilitiesKHR`
	pub fn get_vk_surf_caps(&self) -> &VkSurfaceCapabilitiesKHR {
		&self.surf_caps
	}

	pub fn is_supported_depth_stencil_format(vkcore: &VkCore, vkgpu: VkPhysicalDevice, format: VkFormat) -> Result<bool, VulkanError> {
		let mut format_props: VkFormatProperties = unsafe {MaybeUninit::zeroed().assume_init()};
		vkcore.vkGetPhysicalDeviceFormatProperties(vkgpu, format, &mut format_props)?;
		if (format_props.optimalTilingFeatures & VkFormatFeatureFlagBits::VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT as u32) == VkFormatFeatureFlagBits::VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT as u32 {
			Ok(true)
		} else {
			Ok(false)
		}
	}

	pub fn get_depth_stencil_format(vkcore: &VkCore, physical_device: VkPhysicalDevice) -> Result<VkFormat, VulkanError> {
		const FORMATS: [VkFormat; 3] = [
			VkFormat::VK_FORMAT_D32_SFLOAT_S8_UINT,
			VkFormat::VK_FORMAT_D24_UNORM_S8_UINT,
			VkFormat::VK_FORMAT_D16_UNORM_S8_UINT,
		];
		for fmt in FORMATS.iter() {
			if Self::is_supported_depth_stencil_format(vkcore, physical_device, *fmt)? {
				return Ok(*fmt);
			}
		}
		Err(VulkanError::NoGoodDepthStencilFormat)
	}

	/// Get the current swapchain extent
	pub fn get_swapchain_extent(&self) -> VkExtent2D {
		self.swapchain_extent
	}

	/// Get the currrent present mode
	pub fn get_present_mode(&self) -> VkPresentModeKHR {
		self.present_mode
	}

	/// Get the list of `VulkanSwapchainImage`s
	pub fn get_images(&self) -> &[Arc<Mutex<VulkanSwapchainImage>>] {
		self.images.as_ref()
	}

	/// Get the `VulkanSwapchainImage` by an index
	pub fn get_image(&self, index: usize) -> Arc<Mutex<VulkanSwapchainImage>> {
		self.images[index].clone()
	}

	/// Get the current image index in use
	pub fn get_image_index(&self) -> usize {
		self.cur_image_index as usize
	}

	/// Get the current image index in use
	pub fn get_desired_num_of_swapchain_images(&self) -> usize {
		self.desired_num_of_swapchain_images as usize
	}

	/// Get if the swapchain is VSYNC
	pub fn get_is_vsync(&self) -> bool {
		self.vsync
	}

	/// Get if the swapchain is for VR
	pub fn get_is_vr(&self) -> bool {
		self.is_vr
	}

	/// Acquire the next image, get the new image index
	pub(crate) fn acquire_next_image(&mut self, block: bool) -> Result<usize, VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let device = self.device.get_vk_device();
		let mut cur_image_index = 0u32;
		vkcore.vkAcquireNextImageKHR(device, self.swapchain, if block {u64::MAX} else {0}, self.acquire_semaphore.get_vk_semaphore(), null(), &mut cur_image_index)?;
		self.cur_image_index = cur_image_index;
		let image_lock = self.get_image(cur_image_index as usize);
		swap(&mut self.acquire_semaphore, &mut image_lock.lock().unwrap().acquire_semaphore);
		Ok(cur_image_index as usize)
	}

	/// Enqueue a present command to the queue
	pub(crate) fn queue_present(&self, queue_index: usize, image_index: usize) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let swapchains = [self.swapchain];
		let image_indices = [image_index as u32];
		let wait_semaphores = [self.get_image(image_index).lock().unwrap().release_semaphore.get_vk_semaphore()];
		let present_info = VkPresentInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			pNext: null(),
			waitSemaphoreCount: 1,
			pWaitSemaphores: wait_semaphores.as_ptr(),
			swapchainCount: 1,
			pSwapchains: swapchains.as_ptr(),
			pImageIndices: image_indices.as_ptr(),
			pResults: null_mut(),
		};

		vkcore.vkQueuePresentKHR(self.device.get_vk_queue(queue_index), &present_info)?;
		Ok(())
	}
}

impl Debug for VulkanSwapchain {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanSwapchain")
		.field("surface", &self.surface)
		.field("vsync", &self.vsync)
		.field("is_vr", &self.is_vr)
		.field("surf_caps", &self.surf_caps)
		.field("swapchain", &self.swapchain)
		.field("swapchain_extent", &self.swapchain_extent)
		.field("present_mode", &self.present_mode)
		.field("depth_stencil_format", &self.depth_stencil_format)
		.field("images", &self.images)
		.field("acquire_semaphore", &self.acquire_semaphore)
		.field("cur_image_index", &self.cur_image_index)
		.finish()
	}
}

impl Drop for VulkanSwapchain {
	fn drop(&mut self) {
		let vkcore = self.device.vkcore.clone();
		self.images.clear();
		vkcore.vkDestroySwapchainKHR(self.device.get_vk_device(), self.swapchain, null()).unwrap();
	}
}
