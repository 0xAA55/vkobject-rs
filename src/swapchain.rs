
use crate::prelude::*;
use std::{
	cmp::max,
	fmt::{self, Debug, Formatter},
	mem::{MaybeUninit, transmute, swap},
	ptr::{null, null_mut},
	sync::{Arc, Mutex},
};

/// An image of a swapchain
pub struct VulkanSwapchainImage {
	/// The `VulkanDevice` is the associated device
	pub device: Arc<VulkanDevice>,

	/// The handle to the image
	pub image: Arc<VulkanTexture>,

	/// The depth stencil image
	pub depth_stencil: Arc<VulkanTexture>,

	/// The render target properties
	pub rt_props: Arc<RenderTargetProps>,
}

unsafe impl Send for VulkanSwapchainImage {}

impl VulkanSwapchainImage {
	/// Create the `VulkanSwapchainImage`
	pub(crate) fn new(device: Arc<VulkanDevice>, surface_format: &VkSurfaceFormatKHR, image: VkImage, extent: &VkExtent2D, depth_stencil_format: VkFormat) -> Result<Self, VulkanError> {
		let image = Arc::new(VulkanTexture::new_from_image(device.clone(), image, VulkanTextureType::T2d(*extent), surface_format.format)?);
		let depth_stencil = Arc::new(VulkanTexture::new(
			device.clone(),
			VulkanTextureType::DepthStencil(*extent),
			false,
			depth_stencil_format,
			VkImageUsageFlagBits::VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT as VkImageUsageFlags |
			VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT as VkImageUsageFlags
		)?);
		let attachments = [image.clone(), depth_stencil.clone()];
		let rt_props = Arc::new(RenderTargetProps::new(device.clone(), extent, &attachments)?);
		Ok(Self{
			device,
			image,
			depth_stencil,
			rt_props,
		})
	}

	/// Get the extent of the image
	pub fn get_extent(&self) -> VkExtent2D {
		let extent = self.image.get_extent();
		VkExtent2D {
			width: extent.width,
			height: extent.height,
		}
	}
}

impl Debug for VulkanSwapchainImage {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("VulkanSwapchainImage")
		.field("image", &self.image)
		.field("depth_stencil", &self.depth_stencil)
		.field("rt_props", &self.rt_props)
		.finish()
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
	pub images: Vec<Arc<VulkanSwapchainImage>>,

	/// The semaphores for acquiring new frame image
	acquire_semaphores: Vec<Arc<Mutex<VulkanSemaphore>>>,
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
		let mut images = Vec::<Arc<VulkanSwapchainImage>>::with_capacity(vk_images.len());
		let depth_stencil_format = Self::get_depth_stencil_format(&vkcore, vk_phy_dev)?;
		for vk_image in vk_images.iter() {
			images.push(Arc::new(VulkanSwapchainImage::new(device.clone(), &surface_format, *vk_image, &swapchain_extent, depth_stencil_format)?));
		}
		let mut acquire_semaphores: Vec<Arc<Mutex<VulkanSemaphore>>> = Vec::with_capacity(num_images as usize);
		for _ in 0..num_images {
			acquire_semaphores.push(Arc::new(Mutex::new(VulkanSemaphore::new(device.clone())?)));
		}
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
			acquire_semaphores,
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
	pub fn get_images(&self) -> &[Arc<VulkanSwapchainImage>] {
		self.images.as_ref()
	}

	/// Get the `VulkanSwapchainImage` by an index
	pub fn get_image(&self, index: usize) -> Arc<VulkanSwapchainImage> {
		self.images[index].clone()
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
	pub(crate) fn acquire_next_image(&self, thread_index: usize, timeout: u64) -> Result<usize, VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let vkdevice = self.device.get_vk_device();
		let mut cur_image_index = 0u32;
		let sem = self.acquire_semaphores[thread_index].lock().unwrap().get_vk_semaphore();
		vkcore.vkAcquireNextImageKHR(vkdevice, self.swapchain, timeout, sem, null(), &mut cur_image_index)?;
		let image = self.get_image(cur_image_index as usize);
		swap(&mut *self.acquire_semaphores[thread_index].lock().unwrap(), &mut *image.rt_props.acquire_semaphore.lock().unwrap());
		Ok(cur_image_index as usize)
	}

	/// Enqueue a present command to the queue
	pub(crate) fn queue_present(&self, present_image_index: usize) -> Result<(), VulkanError> {
		let vkcore = self.device.vkcore.clone();
		let swapchains = [self.swapchain];
		let image_indices = [present_image_index as u32];
		let wait_semaphores = [self.get_image(present_image_index).rt_props.release_semaphore.get_vk_semaphore()];
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

		vkcore.vkQueuePresentKHR(self.device.get_vk_queue(), &present_info)?;
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
		.field("desired_num_of_swapchain_images", &self.desired_num_of_swapchain_images)
		.field("images", &self.images)
		.field("acquire_semaphores", &self.acquire_semaphores)
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
