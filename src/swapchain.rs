
use crate::prelude::*;
use std::{
	ffi::c_void,
	fmt::{self, Debug, Formatter},
	mem::{MaybeUninit, transmute},
	ptr::{null, null_mut},
	sync::{Mutex, Arc, Weak},
};

#[derive(Debug)]
pub struct VulkanSwapchainImage {
	pub(crate) ctx: Weak<Mutex<VulkanContext>>,
	pub image: VkImage,
	pub image_view: VkImageView,
	pub acquire_semaphore: VulkanSemaphore,
	pub release_semaphore: VulkanSemaphore,
	pub queue_submit_fence: VulkanFence,
}

unsafe impl Send for VulkanSwapchainImage {}

impl VulkanSwapchainImage {
	pub fn new(vkcore: &VkCore, image: VkImage, surface: &VulkanSurface, device: &VulkanDevice) -> Result<Self, VkError> {
		let vk_image_view_ci = VkImageViewCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			pNext: null(),
			flags: 0,
			image,
			viewType: VkImageViewType::VK_IMAGE_VIEW_TYPE_2D,
			format: surface.get_vk_surface_format().format,
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
		let acquire_semaphore = VulkanSemaphore::new(vkcore, device)?;
		let release_semaphore = VulkanSemaphore::new(vkcore, device)?;
		let queue_submit_fence = VulkanFence::new(vkcore, device)?;
		let vk_device = device.get_vk_device();
		vkcore.vkCreateImageView(vk_device, &vk_image_view_ci, null(), &mut image_view)?;
		Ok(Self{
			ctx: Weak::new(),
			image,
			image_view,
			acquire_semaphore,
			release_semaphore,
			queue_submit_fence,
		})
	}

	pub fn get_vk_image(&self) -> VkImage {
		self.image
	}

	fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.acquire_semaphore.set_ctx(ctx.clone());
		self.release_semaphore.set_ctx(ctx.clone());
		self.queue_submit_fence.set_ctx(ctx.clone());
		self.ctx = ctx;
	}
}

impl Drop for VulkanSwapchainImage {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			let device = ctx.get_vk_device();
			vkcore.vkDestroyImageView(device, self.image_view, null()).unwrap();
		}
	}
}

#[derive(Debug)]
pub struct VulkanSwapchain {
	ctx: Weak<Mutex<VulkanContext>>,
	pub surface: Arc<Mutex<VulkanSurface>>,
	surf_caps: VkSurfaceCapabilitiesKHR,
	swapchain: VkSwapchainKHR,
	swapchain_extent: VkExtent2D,
	present_mode: VkPresentModeKHR,
	pub images: Vec<VulkanSwapchainImage>,
}

unsafe impl Send for VulkanSwapchain {}

impl VulkanSwapchain {
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, surface_: Arc<Mutex<VulkanSurface>>, width: u32, height: u32, vsync: bool, is_vr: bool) -> Result<Self, VkError> {
		let surface = surface_.lock().unwrap();
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
		let mut desired_num_of_swapchain_images = surf_caps.minImageCount + 1;
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
			imageFormat: surface.get_vk_surface_format().format,
			imageColorSpace: surface.get_vk_surface_format().colorSpace,
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
			oldSwapchain: null(),
		};

		let mut swapchain: VkSwapchainKHR = null();
		vkcore.vkCreateSwapchainKHR(vk_device, &swapchain_ci, null(), &mut swapchain)?;
		let mut num_images = 0u32;
		vkcore.vkGetSwapchainImagesKHR(vk_device, swapchain, &mut num_images, null_mut())?;
		let mut vk_images = Vec::<VkImage>::with_capacity(num_images as usize);
		vkcore.vkGetSwapchainImagesKHR(vk_device, swapchain, &mut num_images, vk_images.as_mut_ptr())?;
		unsafe {vk_images.set_len(num_images as usize)};
		let mut images = Vec::<VulkanSwapchainImage>::with_capacity(vk_images.len());
		for vk_image in vk_images.iter() {
			images.push(VulkanSwapchainImage::new(vkcore, *vk_image, &surface, device)?);
		}

		drop(surface);
		Ok(Self {
			ctx: Weak::new(),
			surface: surface_,
			surf_caps,
			swapchain,
			swapchain_extent,
			present_mode,
			images,
		})
	}

	pub(crate) fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		for image in self.images.iter_mut() {
			image.set_ctx(ctx.clone());
		}
		self.ctx = ctx;
	}

	pub fn get_vk_surface(&self) -> VkSurfaceKHR {
		let surface = self.surface.lock().unwrap();
		surface.get_vk_surface()
	}

	pub(crate) fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.swapchain
	}

	pub fn get_vk_surf_caps(&self) -> &VkSurfaceCapabilitiesKHR {
		&self.surf_caps
	}

	pub fn get_swapchain_extent(&self) -> VkExtent2D {
		self.swapchain_extent
	}

	pub fn get_present_mode(&self) -> VkPresentModeKHR {
		self.present_mode
	}

	pub fn get_images(&self) -> &[VulkanSwapchainImage] {
		self.images.as_ref()
	}

	pub fn acquire_next_image(&self, present_complete_semaphore: VkSemaphore, image_index: &mut u32) -> Result<(), VkError> {
		let binding = self.ctx.upgrade().unwrap();
		let ctx = binding.lock().unwrap();
		let vkcore = ctx.get_vkcore();
		let device = ctx.get_vk_device();
		vkcore.vkAcquireNextImageKHR(device, self.swapchain, u64::MAX, present_complete_semaphore, null(), image_index)?;
		Ok(())
	}

	pub fn queue_present(&self, image_index: u32, wait_semaphore: VkSemaphore) -> Result<(), VkError> {
		let binding = self.ctx.upgrade().unwrap();
		let ctx = binding.lock().unwrap();
		let vkcore = ctx.get_vkcore();
		let num_wait_semaphores;
		let wait_semaphores;
		if wait_semaphore != VK_NULL_HANDLE as _ {
			num_wait_semaphores = 1;
			wait_semaphores = &wait_semaphore as *const _;
		} else {
			num_wait_semaphores = 0;
			wait_semaphores = null();
		}
		let present_info = VkPresentInfoKHR {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			pNext: null(),
			waitSemaphoreCount: num_wait_semaphores,
			pWaitSemaphores: wait_semaphores,
			swapchainCount: 1,
			pSwapchains: &self.swapchain as *const _,
			pImageIndices: &image_index as *const _,
			pResults: null_mut(),
		};
		vkcore.vkQueuePresentKHR(ctx.get_vk_queue(), &present_info)?;
		Ok(())
	}
}

impl Drop for VulkanSwapchain {
	fn drop(&mut self) {
		if let Some(binding) = self.ctx.upgrade() {
			let ctx = binding.lock().unwrap();
			let vkcore = ctx.get_vkcore();
			self.images.clear();
			vkcore.vkDestroySwapchainKHR(ctx.get_vk_device(), self.swapchain, null()).unwrap();
		}
	}
}
