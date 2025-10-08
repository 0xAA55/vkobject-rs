
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

/// The Vulkan descriptor pool object
pub mod descpool;

/// The Vulkan context
pub mod context;

/// The buffer object
pub mod buffer;

/// The advanced buffer object that could be used as a vector
pub mod buffervec;

/// The texture object
pub mod texture;

/// The material object
pub mod material;

/// The mesh object
pub mod mesh;

/// The descriptor set properties
pub mod descprops;

/// The pipeline object to wiring up buffers from a mesh, shaders, rendertargets together.
pub mod pipeline;

/// The module for loading OBJ meshes
pub mod wavefrontobj;

extern crate nalgebra_glm as glm;

/// The common things for you to use
pub mod prelude {
	pub use vkcore_rs::*;
	pub use glm::*;
	pub use half::f16;
	pub use struct_iterable::Iterable;
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
	pub use crate::descpool::*;
	pub use crate::context::*;
	pub use crate::buffer::*;
	pub use crate::buffervec::*;
	pub use crate::texture::*;
	pub use crate::material::*;
	pub use crate::mesh::*;
	pub use crate::descprops::*;
	pub use crate::pipeline::*;
	pub use crate::wavefrontobj::*;
	pub use crate::derive_vertex_type;
	pub use crate::derive_uniform_buffer_type;
	pub use crate::derive_storage_buffer_type;
}

#[cfg(test)]
mod tests {
	use glfw::*;
	use crate::prelude::*;
	use std::{
		collections::HashMap,
		ffi::CStr,
		path::PathBuf,
		slice::from_raw_parts_mut,
		sync::{Arc, Mutex, RwLock},
	};

	const TEST_TIME: f64 = 10.0;

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
			static GLFW_LOCK: Mutex<u32> = Mutex::new(0);
			let glfw_lock = GLFW_LOCK.lock().unwrap();
			let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
			glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
			let (mut window, events) = glfw.create_window(width, height, title, window_mode).expect("Failed to create GLFW window.");
			drop(glfw_lock);
			window.set_key_polling(true);
			let ctx = create_vulkan_context(&window, PresentInterval::VSync, 1, false)?;
			println!("{}", unsafe{CStr::from_ptr(ctx.device.get_gpu().properties.deviceName.as_ptr())}.to_str().unwrap());
			println!("{:?}", ctx.device.get_gpu().properties.deviceType);
			Ok(Self {
				glfw,
				window,
				events,
				num_frames: 0,
				ctx,
			})
		}

		pub fn run(&mut self,
			test_time: Option<f64>,
			mut on_render: impl FnMut(&mut VulkanContext, f64) -> Result<(), VulkanError>
		) -> Result<(), VulkanError> {
			let start_time = self.glfw.get_time();
			let mut time_in_sec: u64 = 0;
			let mut num_frames_prev: u64 = 0;
			while !self.window.should_close() {
				let cur_frame_time = self.glfw.get_time();
				let run_time = cur_frame_time - start_time;
				on_render(&mut self.ctx, run_time)?;
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
	unsafe impl Sync for TestInstance {}

	#[test]
	fn basic_test() {
		derive_vertex_type! {
			pub struct VertexType {
				pub position: Vec2,
			}
		}
		derive_uniform_buffer_type! {
			pub struct UniformInput {
				resolution: Vec3,
				time: f32,
			}
		}
		struct Resources {
			uniform_input: Arc<dyn GenericUniformBuffer>,
			pipeline: Pipeline,
		}

		impl Resources {
			pub fn new(ctx: &mut VulkanContext) -> Result<Self, VulkanError> {
				let device = ctx.device.clone();
				let draw_shaders = Arc::new(DrawShaders::new(
					Arc::new(VulkanShader::new_from_source_file_or_cache(device.clone(), ShaderSourcePath::VertexShader(PathBuf::from("shaders/test.vsh")), false, "main", OptimizationLevel::Performance, false)?),
					None,
					None,
					None,
					Arc::new(VulkanShader::new_from_source_file_or_cache(device.clone(), ShaderSourcePath::FragmentShader(PathBuf::from("shaders/test.fsh")), false, "main", OptimizationLevel::Performance, false)?),
				));
				let uniform_input: Arc<dyn GenericUniformBuffer> = Arc::new(UniformBuffer::<UniformInput>::new(device.clone())?);
				let desc_prop = vec![uniform_input.clone()];
				let desc_props: HashMap<u32, HashMap<u32, Arc<DescriptorProp>>> = [(0, [(0, Arc::new(DescriptorProp::UniformBuffers(desc_prop)))].into_iter().collect())].into_iter().collect();
				let desc_props = Arc::new(DescriptorProps::new(desc_props));
				let pool_in_use = ctx.cmdpools[0].use_pool(None)?;
				let vertices_data = vec![
					VertexType {
						position: Vec2::new(-1.0, -1.0),
					},
					VertexType {
						position: Vec2::new( 1.0, -1.0),
					},
					VertexType {
						position: Vec2::new(-1.0,  1.0),
					},
					VertexType {
						position: Vec2::new( 1.0,  1.0),
					},
				];
				let vertices = Arc::new(RwLock::new(BufferWithType::new(device.clone(), &vertices_data, pool_in_use.cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?));
				let mesh = Arc::new(GenericMeshWithMaterial::new(Arc::new(Mesh::new(VkPrimitiveTopology::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, vertices, buffer_unused(), buffer_unused(), buffer_unused())), "", None));
				mesh.geometry.flush(pool_in_use.cmdbuf)?;
				drop(pool_in_use);
				ctx.cmdpools[0].wait_for_submit(u64::MAX)?;
				mesh.geometry.discard_staging_buffers();
				let pipeline = ctx.create_pipeline_builder(mesh, draw_shaders, desc_props.clone())?
				.set_cull_mode(VkCullModeFlagBits::VK_CULL_MODE_NONE as VkCullModeFlags)
				.set_depth_test(false)
				.set_depth_write(false)
				.build()?;
				Ok(Self {
					uniform_input,
					pipeline,
				})
			}

			pub fn draw(&self, ctx: &mut VulkanContext, run_time: f64) -> Result<(), VulkanError> {
				let scene = ctx.begin_scene(0, None)?;
				let cmdbuf = scene.get_cmdbuf();
				let extent = scene.get_rendertarget_extent();

				let ui_data = unsafe {from_raw_parts_mut(self.uniform_input.get_staging_buffer_address() as *mut UniformInput, 1)};
				ui_data[0] = UniformInput {
					resolution: Vec3::new(extent.width as f32, extent.height as f32, 1.0),
					time: run_time as f32,
				};
				self.uniform_input.flush(cmdbuf)?;

				scene.set_viewport_swapchain(0.0, 1.0)?;
				scene.set_scissor_swapchain()?;
				scene.begin_renderpass(Vec4::new(0.0, 0.0, 0.2, 1.0), 1.0, 0)?;
				self.pipeline.draw(cmdbuf)?;
				scene.end_renderpass()?;
				scene.finish();
				Ok(())
			}
		}

		let mut inst = Box::new(TestInstance::new(1024, 768, "Vulkan test", glfw::WindowMode::Windowed).unwrap());
		let resources = Resources::new(&mut inst.ctx).unwrap();
		inst.run(Some(TEST_TIME),
		|ctx: &mut VulkanContext, run_time: f64| -> Result<(), VulkanError> {
			resources.draw(ctx, run_time)
		}).unwrap();
	}

	#[test]
	fn avocado() {
		derive_vertex_type! {
			pub struct InstanceType {
				pub transform: Mat4,
			}
		}
		derive_uniform_buffer_type! {
			pub struct UniformInputScene {
				light_dir: Vec3,
				light_color: Vec3,
				view_matrix: Mat4,
				proj_matrix: Mat4,
			}
		}
		derive_uniform_buffer_type! {
			pub struct UniformInputObject {
				obj_color: Vec3,
				obj_specular: Vec4,
			}
		}
		struct Resources {
			uniform_input_scene: Arc<dyn GenericUniformBuffer>,
			uniform_input_object: Arc<dyn GenericUniformBuffer>,
			object: GenericMeshSet<InstanceType>,
			pipelines: HashMap<String, Pipeline>,
		}

		impl Resources {
			pub fn new(ctx: &mut VulkanContext) -> Result<Self, VulkanError> {
				let device = ctx.device.clone();
				let draw_shaders = Arc::new(DrawShaders::new(
					Arc::new(VulkanShader::new_from_source_file_or_cache(device.clone(), ShaderSourcePath::VertexShader(PathBuf::from("shaders/objdisp.vsh")), false, "main", OptimizationLevel::Performance, false)?),
					None,
					None,
					None,
					Arc::new(VulkanShader::new_from_source_file_or_cache(device.clone(), ShaderSourcePath::FragmentShader(PathBuf::from("shaders/objdisp.fsh")), false, "main", OptimizationLevel::Performance, false)?),
				));
				let pool_in_use = ctx.cmdpools[0].use_pool(None)?;
				let object = GenericMeshSet::create_meshset_from_obj_file::<f32, _>(device.clone(), "assets/testobj/avocado.obj", pool_in_use.cmdbuf, Some(&[InstanceType {transform: Mat4::identity()}]))?;
				let uniform_input_scene: Arc<dyn GenericUniformBuffer> = Arc::new(UniformBuffer::<UniformInputScene>::new(device.clone())?);
				let uniform_input_object: Arc<dyn GenericUniformBuffer> = Arc::new(UniformBuffer::<UniformInputObject>::new(device.clone())?);
				let mut desc_props_set_0: HashMap<u32, Arc<DescriptorProp>> = [
					(0, Arc::new(DescriptorProp::UniformBuffers(vec![uniform_input_scene.clone()]))),
					(1, Arc::new(DescriptorProp::UniformBuffers(vec![uniform_input_object.clone()]))),
				].into_iter().collect();
				let mut pipelines: HashMap<String, Pipeline> = HashMap::with_capacity(object.meshset.len());
				for mesh in object.meshset.values() {
					if let Some(material) = &mesh.material {
						if let Some(diffuse) = material.get_diffuse() {
							if let MaterialComponent::Texture(texture) = diffuse {
								texture.prepare_for_sample(pool_in_use.cmdbuf)?;
								let texture_input_albedo = TextureForSample {
									texture: texture.clone(),
									sampler: Arc::new(VulkanSampler::new_linear(device.clone(), true, false)?),
								};
								desc_props_set_0.insert(2, Arc::new(DescriptorProp::Images(vec![texture_input_albedo])));
							}
						}
						if let Some(normal) = material.get_normal() {
							if let MaterialComponent::Texture(texture) = normal {
								texture.prepare_for_sample(pool_in_use.cmdbuf)?;
								let texture_input_normal = TextureForSample {
									texture: texture.clone(),
									sampler: Arc::new(VulkanSampler::new_linear(device.clone(), true, false)?),
								};
								desc_props_set_0.insert(3, Arc::new(DescriptorProp::Images(vec![texture_input_normal])));
							}
						}
					}
				}
				drop(pool_in_use);
				ctx.cmdpools[0].wait_for_submit(u64::MAX)?;
				object.discard_staging_buffers();
				let desc_props_sets: HashMap<u32, HashMap<u32, Arc<DescriptorProp>>> = [(0, desc_props_set_0)].into_iter().collect();
				let desc_props = Arc::new(DescriptorProps::new(desc_props_sets));
				for (matname, mesh) in object.meshset.iter() {
					let pipeline = ctx.create_pipeline_builder(mesh.clone(), draw_shaders.clone(), desc_props.clone())?
					.set_cull_mode(VkCullModeFlagBits::VK_CULL_MODE_NONE as VkCullModeFlags)
					.set_depth_test(true)
					.set_depth_write(true)
					.build()?;
					pipelines.insert(matname.clone(), pipeline);
				}
				Ok(Self {
					uniform_input_scene,
					uniform_input_object,
					object,
					pipelines,
				})
			}

			pub fn draw(&self, ctx: &mut VulkanContext, run_time: f64) -> Result<(), VulkanError> {
				let scene = ctx.begin_scene(0, None)?;
				let cmdbuf = scene.get_cmdbuf();
				let extent = scene.get_rendertarget_extent();

				let view_matrix = {
					let eye = glm::vec3(0.0, 20.0, 50.0);
					let center = glm::vec3(0.0, 0.0, 0.0);
					let up = glm::vec3(0.0, 1.0, 0.0);
					glm::look_at(&eye, &center, &up)
				};

				let proj_matrix = {
					let fovy = pi::<f32>() / 3.0;
					let aspect = extent.width as f32 / extent.height as f32;
					perspective_rh_zo(fovy, aspect, 0.1, 1000.0)
				};

				let ui_data = unsafe {from_raw_parts_mut(self.uniform_input_scene.get_staging_buffer_address() as *mut UniformInputScene, 1)};
				ui_data[0] = UniformInputScene {
					light_dir: normalize(&Vec3::new(0.2, -1.0, 0.2)),
					light_color: Vec3::new(1.0, 1.0, 1.0),
					view_matrix,
					proj_matrix,
				};
				let ui_data = unsafe {from_raw_parts_mut(self.uniform_input_scene.get_staging_buffer_address() as *mut UniformInputObject, 1)};
				ui_data[0] = UniformInputObject {
					obj_color: Vec3::new(1.0, 1.0, 1.0),
					obj_specular: Vec4::new(0.0, 0.0, 0.0, 1.0),
				};
				self.uniform_input_scene.flush(cmdbuf)?;
				self.uniform_input_object.flush(cmdbuf)?;

				scene.set_viewport_swapchain(0.0, 1.0)?;
				scene.set_scissor_swapchain()?;
				for pipeline in self.pipelines.values() {
					scene.begin_renderpass(Vec4::new(0.0, 0.0, 0.2, 1.0), 1.0, 0)?;
					pipeline.draw(cmdbuf)?;
					scene.end_renderpass()?;
				}
				scene.finish();
				Ok(())
			}
		}

		let mut inst = Box::new(TestInstance::new(1024, 768, "Vulkan avocado test", glfw::WindowMode::Windowed).unwrap());
		let resources = Resources::new(&mut inst.ctx).unwrap();
		inst.run(Some(TEST_TIME),
		|ctx: &mut VulkanContext, run_time: f64| -> Result<(), VulkanError> {
			resources.draw(ctx, run_time)
		}).unwrap();
	}
}
