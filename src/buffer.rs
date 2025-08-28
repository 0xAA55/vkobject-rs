
use crate::prelude::*;
use std::{
	cell::RefCell,
	sync::Weak,
};

#[derive(Debug)]
pub struct Buffer {
	/// The `VulkanContext` that helps to manage the resources of the buffer
	ctx: Weak<RefCell<VulkanContext>>,

	/// The handle to the device memory
	memory: VkDeviceMemory,

	/// The handle to the buffer
	buffer: VkBuffer,
}

impl Buffer {
	/// Get the buffer memory
	pub fn get_memory(&self) -> VkDeviceMemory {
		self.memory
	}

	/// Get the buffer
	pub fn get_buffer(&self) -> VkBuffer {
		self.buffer
	}
}
