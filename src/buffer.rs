
use crate::prelude::*;
use std::{
	cell::RefCell,
	sync::{Arc, Weak},
};

#[derive(Debug)]
pub struct Buffer {
	pub states: Weak<RefCell<VulkanStates>>,
	memory: VkDeviceMemory,
	buffer: VkBuffer,
}

impl Buffer {
	/// Get the buffer memory
	pub fn get_memory(&self) -> VkDeviceMemory {
		self.memory
	}

	/// Get the buffer
	pub fn get_buffer(&self) ->VkBuffer {
		self.buffer
	}
}
