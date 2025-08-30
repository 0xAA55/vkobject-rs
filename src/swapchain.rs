
use crate::prelude::*;
use std::{
	fmt::Debug,
	mem::{MaybeUninit, transmute, swap},
	ptr::{null, null_mut},
	sync::{Mutex, Arc, Weak},
};

/// An image of a swap chain
#[derive(Debug)]
pub struct VulkanSwapchainImage {
	/// The `VulkanContext` that helps to manage the resources of the swapchain image
	pub(crate) ctx: Weak<Mutex<VulkanContext>>,

	/// The handle to the image
	image: VkImage,

	/// The handle to the image view
	image_view: VkImageView,

	/// The extent of the image
	extent: VkExtent2D,

	/// The semaphore to acquire the image for rendering
	pub acquire_semaphore: VulkanSemaphore,

	/// The semaphore for the image on release
	pub release_semaphore: VulkanSemaphore,

	/// The fence of submitting commands to a queue
	pub queue_submit_fence: VulkanFence,
}

unsafe impl Send for VulkanSwapchainImage {}

impl VulkanSwapchainImage {
	/// Create the `VulkanSwapchainImage`
	pub fn new_(vkcore: &VkCore, device: VkDevice, surface: &VulkanSurface, image: VkImage, extent: &VkExtent2D) -> Result<Self, VulkanError> {
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
		let acquire_semaphore = VulkanSemaphore::new_(vkcore, device)?;
		let release_semaphore = VulkanSemaphore::new_(vkcore, device)?;
		let queue_submit_fence = VulkanFence::new_(vkcore, device)?;
		vkcore.vkCreateImageView(device, &vk_image_view_ci, null(), &mut image_view)?;
		Ok(Self{
			ctx: Weak::new(),
			image,
			image_view,
			extent: *extent,
			acquire_semaphore,
			release_semaphore,
			queue_submit_fence,
		})
	}

	/// Create the `VulkanSwapchainImage`
	pub fn new(ctx: Arc<Mutex<VulkanContext>>, image: VkImage, extent: &VkExtent2D) -> Result<Self, VulkanError> {
		let ctx_lock = ctx.lock().unwrap();
		let vkcore = ctx_lock.vkcore.clone();
		let surface = ctx_lock.surface.lock().unwrap();
		let vkdevice = ctx_lock.get_vk_device();
		let mut ret = Self::new_(&vkcore, vkdevice, &*surface, image, extent)?;
		drop(surface);
		drop(ctx_lock);
		ret.set_ctx(Arc::downgrade(&ctx));
		Ok(ret)
	}

	/// Get the `VkImage`
	pub(crate) fn get_vk_image(&self) -> VkImage {
		self.image
	}

	/// Get the `VkImageView`
	pub(crate) fn get_vk_image_view(&self) -> VkImageView {
		self.image_view
	}

	/// Set the `VulkanContext` if it hasn't provided previously
	pub(crate) fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.acquire_semaphore.set_ctx(ctx.clone());
		self.release_semaphore.set_ctx(ctx.clone());
		self.queue_submit_fence.set_ctx(ctx.clone());
		self.ctx = ctx;
	}

	/// Get the extent of the image
	pub fn get_extent(&self) -> &VkExtent2D {
		&self.extent
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

/// A swapchain for presenting frames to the window surface.
#[derive(Debug)]
pub struct VulkanSwapchain {
	/// The `VulkanContext` that helps to manage the resources of the swapchain
	ctx: Weak<Mutex<VulkanContext>>,

	/// The surface that helps the creation of the swapchain
	pub surface: Arc<Mutex<VulkanSurface>>,

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

	/// The swapchain images
	pub images: Vec<VulkanSwapchainImage>,

	/// The semaphore for acquiring new frame image
	acquire_semaphore: VulkanSemaphore,

	/// The current image index in use
	cur_image_index: u32,
}

unsafe impl Send for VulkanSwapchain {}

impl VulkanSwapchain {
	/// Create the `VulkanSwapchain`
	pub fn new(vkcore: &VkCore, device: &VulkanDevice, surface_: Arc<Mutex<VulkanSurface>>, width: u32, height: u32, vsync: bool, is_vr: bool, old_swapchain: Option<VkSwapchainKHR>) -> Result<Self, VulkanError> {
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
			oldSwapchain: match old_swapchain {
				Some(chain) => chain,
				None => null(),
			},
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
			images.push(VulkanSwapchainImage::new_(vkcore, vk_device, &surface, *vk_image, &swapchain_extent)?);
		}

		drop(surface);
		Ok(Self {
			ctx: Weak::new(),
			surface: surface_,
			vsync,
			is_vr,
			surf_caps,
			swapchain,
			swapchain_extent,
			present_mode,
			images,
			acquire_semaphore: VulkanSemaphore::new_(vkcore, vk_device)?,
			cur_image_index: 0,
		})
	}

	/// Set the `VulkanContext` if it hasn't provided previously
	pub(crate) fn set_ctx(&mut self, ctx: Weak<Mutex<VulkanContext>>) {
		self.acquire_semaphore.set_ctx(ctx.clone());
		for image in self.images.iter_mut() {
			image.set_ctx(ctx.clone());
		}
		self.ctx = ctx;
	}

	/// Get the `VkSurfaceKHR`
	pub fn get_vk_surface(&self) -> VkSurfaceKHR {
		let surface = self.surface.lock().unwrap();
		surface.get_vk_surface()
	}

	/// Get the `VkSwapchainKHR`
	pub(crate) fn get_vk_swapchain(&self) -> VkSwapchainKHR {
		self.swapchain
	}

	/// Get the `VkSurfaceCapabilitiesKHR`
	pub fn get_vk_surf_caps(&self) -> &VkSurfaceCapabilitiesKHR {
		&self.surf_caps
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
	pub fn get_images(&self) -> &[VulkanSwapchainImage] {
		self.images.as_ref()
	}

	/// Get the `VulkanSwapchainImage` by an index
	pub fn get_image(&self, index: usize) -> &VulkanSwapchainImage {
		&self.images[index]
	}

	/// Get the current `VulkanSwapchainImage` in use
	pub fn get_cur_image(&self) -> &VulkanSwapchainImage {
		&self.images[self.cur_image_index as usize]
	}

	/// Get the current image index in use
	pub fn get_image_index(&self) -> u32 {
		self.cur_image_index
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
	pub(crate) fn acquire_next_image(&mut self) -> Result<usize, VulkanError> {
		let binding = self.ctx.upgrade().unwrap();
		let ctx = binding.lock().unwrap();
		let vkcore = ctx.get_vkcore();
		let device = ctx.get_vk_device();
		let mut cur_image_index = 0;
		vkcore.vkAcquireNextImageKHR(device, self.swapchain, u64::MAX, self.acquire_semaphore.get_vk_semaphore(), null(), &mut cur_image_index)?;
		self.cur_image_index = cur_image_index;
		swap(&mut self.acquire_semaphore, &mut self.images[cur_image_index as usize].acquire_semaphore);
		Ok(cur_image_index as usize)
	}

	/// Enqueue a present command to the queue
	pub(crate) fn queue_present(&self, queue_index: usize) -> Result<(), VulkanError> {
		let binding = self.ctx.upgrade().unwrap();
		let ctx = binding.lock().unwrap();
		let vkcore = ctx.get_vkcore();
		let swapchains = [self.swapchain];
		let image_indices = [self.cur_image_index];
		let wait_semaphores = [self.get_cur_image().release_semaphore.get_vk_semaphore()];
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

		vkcore.vkQueuePresentKHR(*ctx.get_vk_queue(queue_index), &present_info)?;
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
