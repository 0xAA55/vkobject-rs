
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

/// The Vulkan renderpass
pub mod renderpass;

/// The Vulkan framebuffer
pub mod framebuffer;

/// The render target
pub mod rendertarget;

/// The Vulkan swapchain
pub mod swapchain;

/// The Vulkan command pool
pub mod cmdpool;

/// The Vulkan shader
pub mod shader;

/// The Vulkan context
pub mod context;

/// The buffer object
pub mod buffer;

/// The advanced buffer object that could be used as a vector
pub mod buffervec;

/// The texture object
pub mod texture;

extern crate nalgebra_glm as glm;

/// The common things for you to use
pub mod prelude {
	pub use vkcore_rs::*;
	pub use glm::*;
	pub use half::f16;
	pub use crate::common::*;
	pub use crate::init::*;
	pub use crate::basics::*;
	pub use crate::device::*;
	pub use crate::surface::*;
	pub use crate::renderpass::*;
	pub use crate::framebuffer::*;
	pub use crate::rendertarget::*;
	pub use crate::swapchain::*;
	pub use crate::cmdpool::*;
	pub use crate::shader::*;
	pub use crate::context::*;
	pub use crate::buffer::*;
	pub use crate::buffervec::*;
	pub use crate::texture::*;
}

#[cfg(test)]
mod tests {
	use glfw::*;
	use crate::prelude::*;

	const TEST_TIME: f64 = 10.0;
	const MAX_CONCURRENT_FRAMES: usize = 3;

	#[derive(Debug)]
	pub struct TestInstance {
		pub ctx: VulkanContext,
		pub window: PWindow,
		pub events: GlfwReceiver<(f64, WindowEvent)>,
		pub glfw: Glfw,
		pub num_frames: u64,
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
				window,
				events,
				num_frames: 0,
				ctx,
			})
		}

		pub fn run(&mut self, test_time: Option<f64>) -> Result<(), VulkanError> {
			let start_time = self.glfw.get_time();
			let mut time_in_sec: u64 = 0;
			let mut num_frames_prev: u64 = 0;
			while !self.window.should_close() {
				let cur_frame_time = self.glfw.get_time();
				let run_time = cur_frame_time - start_time;
				let scene = self.ctx.begin_scene(0, None)?;

				scene.set_viewport_swapchain(0.0, 1.0)?;
				scene.set_scissor_swapchain()?;
				let r = (cur_frame_time.sin() * 0.5 + 0.5) as f32;
				let g = (cur_frame_time.cos() * 0.5 + 0.5) as f32;
				let b = ((cur_frame_time * 1.5).sin() * 0.5 + 0.5) as f32;
				scene.clear(Vec4::new(r, g, b, 1.0), 1.0, 0)?;

				drop(scene);
				self.num_frames += 1;

				let new_time_in_sec = run_time.floor() as u64;
				if new_time_in_sec > time_in_sec {
					let fps = self.num_frames - num_frames_prev;
					println!("FPS: {fps}\tat {new_time_in_sec}s");
					time_in_sec = new_time_in_sec;
					num_frames_prev = self.num_frames;
				}

				self.glfw.poll_events();
				for (_, event) in glfw::flush_messages(&self.events) {
					match event {
						glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
							self.window.set_should_close(true);
						}
						_ => {}
					}
				}
				if let Some(test_time) = test_time {
					if run_time >= test_time {
						self.window.set_should_close(true);
					}
				}
			}
			println!("End of the test");
			Ok(())
		}
	}

	unsafe impl Send for TestInstance {}

	#[test]
	fn test() {
		let mut inst = Box::new(TestInstance::new(1024, 768, "GLFW Window", glfw::WindowMode::Windowed).unwrap());
		inst.run(Some(TEST_TIME)).unwrap();
	}
}
