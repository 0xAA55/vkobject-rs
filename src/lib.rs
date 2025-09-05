
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

/// The common helper library
pub mod common;

/// The VkCore initializer
pub mod init;

/// The Vulkan basics
pub mod basics;

/// The Vulkan device
pub mod device;

/// The Vulkan surface
pub mod surface;

/// The Vulkan swapchain
pub mod swapchain;

/// The Vulkan framebuffer
pub mod framebuffer;

/// The Vulkan renderpass
pub mod renderpass;

/// The Vulkan command pool
pub mod cmdpool;

/// The Vulkan context
pub mod context;

/// The buffer object
pub mod buffer;

/// The texture object
pub mod texture;

extern crate nalgebra_glm as glm;

/// The common things for you to use
pub mod prelude {
	pub use vkcore_rs::*;
	pub use glm::*;
	pub use crate::common::*;
	pub use crate::init::*;
	pub use crate::basics::*;
	pub use crate::device::*;
	pub use crate::surface::*;
	pub use crate::swapchain::*;
	pub use crate::framebuffer::*;
	pub use crate::renderpass::*;
	pub use crate::cmdpool::*;
	pub use crate::context::*;
	pub use crate::buffer::*;
	pub use crate::texture::*;
}

#[cfg(test)]
mod tests {
	use glfw::*;
	use crate::prelude::*;
	use std::{
		sync::{
			atomic::{AtomicU64, Ordering},
			Mutex,
		},
		thread::spawn,
	};

	const TEST_TIME: f64 = 10.0;
	const MAX_CONCURRENT_FRAMES: usize = 3;

	#[derive(Debug)]
	pub struct TestInstance {
		pub ctx: VulkanContext,
		pub window: Mutex<PWindow>,
		pub events: GlfwReceiver<(f64, WindowEvent)>,
		pub glfw: Glfw,
		pub num_frames: AtomicU64,
	}

	impl TestInstance {
		pub fn new(width: u32, height: u32, title: &str, window_mode: glfw::WindowMode) -> Result<Self, VulkanError> {
			let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
			glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
			let (mut window, events) = glfw.create_window(width, height, title, window_mode).expect("Failed to create GLFW window.");
			window.set_key_polling(true);
			let ctx = create_vulkan_context(&window, true, MAX_CONCURRENT_FRAMES, false)?;
			Ok(Self {
				glfw,
				window: Mutex::new(window),
				events,
				num_frames: AtomicU64::new(0),
				ctx,
			})
		}

		fn render(&mut self, pool_index: usize) -> Result<(), VulkanError> {
			loop {
				let lock = self.window.lock().unwrap();
				if lock.should_close() {
					break;
				}
				drop(lock);
				self.ctx.on_resize()?;
				let frame = self.ctx.begin_frame(pool_index, true)?;

				drop(frame);
				self.num_frames.fetch_add(1, Ordering::Relaxed);
			}
			Ok(())
		}

		pub fn run(&mut self, test_time: Option<f64>) {
			let mut renderers = Vec::with_capacity(MAX_CONCURRENT_FRAMES);
			for i in 0..MAX_CONCURRENT_FRAMES {
				let ptr = self as *mut Self as usize;
				renderers.push(spawn(move || {
					let this = unsafe {&mut *(ptr as *mut Self)};
					this.render(i).unwrap();
				}));
			}

			let start_time = self.glfw.get_time();
			let mut time_in_sec: u64 = 0;
			let mut num_frames_prev: u64 = 0;
			loop {
				let cur_frame_time = self.glfw.get_time();
				let run_time = cur_frame_time - start_time;
				let lock = self.window.lock().unwrap();
				if lock.should_close() {
					break;
				}
				drop(lock);

				let new_time_in_sec = run_time.floor() as u64;
				if new_time_in_sec > time_in_sec {
					let frames = self.num_frames.fetch_add(0, Ordering::Relaxed);
					let fps = frames - num_frames_prev;
					println!("FPS: {fps}\tat {new_time_in_sec}s");
					time_in_sec = new_time_in_sec;
					num_frames_prev = frames;
				}

				self.glfw.poll_events();
				for (_, event) in glfw::flush_messages(&self.events) {
					match event {
						glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
							let mut lock = self.window.lock().unwrap();
							lock.set_should_close(true);
						}
						_ => {}
					}
				}
				if let Some(test_time) = test_time {
					if run_time >= test_time {
						let mut lock = self.window.lock().unwrap();
						lock.set_should_close(true);
					}
				}
			}
			println!("End of the test");
			loop {
				if let Some(h) = renderers.pop() {
					h.join().unwrap();
				} else {
					break;
				}
			}
		}
	}

	unsafe impl Send for TestInstance {}

	#[test]
	fn test() {
		let mut inst = Box::new(TestInstance::new(1024, 768, "GLFW Window", glfw::WindowMode::Windowed).unwrap());
		inst.run(Some(TEST_TIME));
	}
}
