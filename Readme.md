# Vulkan Object Wrapper: another Rust renderer engine implementation

## 语言｜language

[简体中文](Readme-CN.md) | Chinglish

## Usage

### Add this crate to your project
```cmd
cargo add vkobject-rs
```

## Rendering strategy

1. Create the `VulkanContext`
- If you want to use **vkobject-rs** with GLFW, the simplest way is to use `create_vulkan_context()` with your GLFW window.

2. Build pipelines for you to draw.
- A pipeline is an object that gathers all of the data from all of your buffers into a shader, runs the shader, and then wires the output to the render target attachments.
- To build a pipeline, you will have to create these things:
	1. Mesh. A mesh describes the polygon you want to render.
	2. Shaders. The shaders define how to process polygons to make pixels and put the pixels into which target attachments.
	3. The data for the shader's inputs. A uniform buffer, storage buffer, push constants, textures, or texel buffers could do this.

3. Draw with the pipeline.
- See the implementation of `pub fn draw()` in the **Example code** section.

4. Resource clean-ups were done automatically when your objects go out of scope, appreciated by the RAII rule of the Rust language.

## The mainly used objects

### Buffers: to hold your polygons, draw instances, draw commands, and shader storages

There are some kinds of buffers in this crate:
- `BufferWithType<T>`: A wrapper for `Buffer`, mainly for the data that won't be modified frequently, so its staging buffer could be discarded.
- `BufferVec<T>`: A wrapper for `Buffer`, provides an interface that is like a `Vec<T>`, call `flush()` could upload data to GPU with a command buffer.
	- The `flush()` will only upload the modified part of the data to the GPU. The data updation is incremental, minimizing the bandwidth usage of CPU-GPU data transfer.
	- The staging buffer of this buffer is not discardable.
- `UniformBuffer<T>`: A wrapper for `Buffer`, whose data on the CPU side could be dereferenced into a structure, and you can modify the structure members freely.
	- The `flush()` will upload the whole structure to the GPU side.
	- This buffer is commonly used for shader inputs.
- `GenericUniformBuffer`: A wrapper for `UniformBuffer<T>` that erases the generic type `<T>`.
- `StorageBuffer<T>`: A wrapper for `Buffer` and the usage is the same as `UniformBuffer<T>` except it's for the shader's storage buffer inputs.
	- The shaders could modify the storage buffers freely, while they can't modify uniform buffers.
- `GenericStorageBuffer`: A wrapper for `StorageBuffer<T>` that erases the generic type `<T>`.
- `StagingBuffer`: The staging buffer that's totally transparent to your CPU, you can have its data pointer and modify the data for it to upload to the GPU.
	- Safety: You should have to know how to manipulate raw pointers correctly.
- `Buffer`: The buffer that's transparent to the GPU, and has its own staging buffer for uploading data to the GPU.
	- Transfer data to its staging buffer, then call `upload_staging_buffer()` to enqueue an upload command into a command buffer.
	- After the upload command is executed, the data is transferred into the GPU, then you call `discard_staging_buffer()` to save some system memory if you wish.
	- This thing is raw; you don't want to use this.
- `VulkanBuffer`: The most low-level buffer wrapping object, the `Buffer` and `StagingBuffer` are implemented by using this object.
	- This thing is super raw; you don't want to use this.

### Mesh: to hold your polygons, draw instances, draw commands

A mesh has 4 buffers. According to the usage, they are:
- vertex buffer
- index buffer (optional)
- instance buffer (optional)
- indirect draw command buffer (optional)

The buffers for a mesh have two types:
- For static draw usage, there is `BufferWithType<T>`
	- The data in the buffer is once initialized, and then never changes.
- For dynamic update usage, there is `BufferVec<T>`
	- You can modify its data frequently like a `Vec<T>`, then call `flush()` to apply changes to the GPU buffer.

```rust
#[derive(Debug, Clone)]
pub struct Mesh<BV, V, BE, E, BI, I, BC, C>
where
	BV: BufferForDraw<V>,
	BE: BufferForDraw<E>,
	BI: BufferForDraw<I>,
	BC: BufferForDraw<C>,
	V: BufferVecStructItem,
	E: BufferVecItem + 'static,
	I: BufferVecStructItem,
	C: BufferVecStructItem {
	pub primitive_type: VkPrimitiveTopology,
	pub vertices: BV,
	pub indices: Option<BE>,
	pub instances: Option<BI>,
	pub commands: Option<BC>,
	vertex_type: V,
	element_type: E,
	instance_type: I,
	command_type: C,
}

/// If a buffer you don't need, use this for your buffer item type
#[derive(Default, Debug, Clone, Copy, Iterable)]
pub struct UnusedBufferItem {}

/// If a buffer you don't need, use this for your buffer type
pub type UnusedBufferType = BufferWithType<UnusedBufferItem>;

/// Use this function to create an unused buffer type
pub fn buffer_unused() -> Option<UnusedBufferType> {
	None
}

impl<BV, V, BE, E, BI, I, BC, C> Mesh<BV, V, BE, E, BI, I, BC, C>
where
	BV: BufferForDraw<V>,
	BE: BufferForDraw<E>,
	BI: BufferForDraw<I>,
	BC: BufferForDraw<C>,
	V: BufferVecStructItem,
	E: BufferVecItem + 'static,
	I: BufferVecStructItem,
	C: BufferVecStructItem {
	/// Create the mesh from the buffers
	pub fn new(primitive_type: VkPrimitiveTopology, vertices: BV, indices: Option<BE>, instances: Option<BI>, commands: Option<BC>) -> Self {
		Self {
			primitive_type,
			vertices,
			indices,
			instances,
			commands,
			vertex_type: V::default(),
			element_type: E::default(),
			instance_type: I::default(),
			command_type: C::default(),
		}
	}

	/// Upload staging buffers to GPU
	pub fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		filter_no_staging_buffer(self.vertices.flush(cmdbuf))?;
		if let Some(ref mut indices) = self.indices {filter_no_staging_buffer(indices.flush(cmdbuf))?;}
		if let Some(ref mut instances) = self.instances {filter_no_staging_buffer(instances.flush(cmdbuf))?;}
		if let Some(ref mut commands) = self.commands {filter_no_staging_buffer(commands.flush(cmdbuf))?;}
		Ok(())
	}

	/// Discard staging buffers if the data will never be modified.
	pub fn discard_staging_buffers(&mut self) {
		self.vertices.discard_staging_buffer();
		if let Some(ref mut indices) = self.indices {indices.discard_staging_buffer();}
		if let Some(ref mut instances) = self.instances {instances.discard_staging_buffer();}
		if let Some(ref mut commands) = self.commands {commands.discard_staging_buffer();}
	}
}
```

### Shaders

The shaders in this crate can compile GLSL or HLSL code to SPIR-V intermediate language. Also, they could be loaded from a binary file instead of the source code.

```rust
let draw_shaders = Arc::new(DrawShaders::new(
	Arc::new(VulkanShader::new_from_source_file_or_cache(device.clone(), ShaderSourcePath::VertexShader(PathBuf::from("shaders/test.vsh")), false, "main", OptimizationLevel::Performance, false)?),
	None,
	None,
	None,
	Arc::new(VulkanShader::new_from_source_file_or_cache(device.clone(), ShaderSourcePath::FragmentShader(PathBuf::from("shaders/test.fsh")), false, "main", OptimizationLevel::Performance, false)?),
));
```

### Texture

The `VulkanTexture` is the wrapper for you to use textures.

### DescriptorProps

The descriptor properties are for the shader inputs; they define which descriptor set and binding has a uniform buffer, or texture, samplers, etc.
- The shader inputs were made as `Vec<T>` since this could help to provide data for array-type inputs of the shaders.
- For a single variable input, simply providing one element of the array could work.

```rust
/// The properties for the descriptor set
#[derive(Debug)]
pub enum DescriptorProp {
	/// The props for the samplers
	Samplers(Vec<Arc<VulkanSampler>>),

	/// The props for the image
	Images(Vec<TextureForSample>),

	/// The props for the storage buffer
	StorageBuffers(Vec<Arc<dyn GenericStorageBuffer>>),

	/// The props for the uniform buffers
	UniformBuffers(Vec<Arc<dyn GenericUniformBuffer>>),

	/// The props for the storage texel buffer
	StorageTexelBuffers(Vec<VulkanBufferView>),

	/// The props for the uniform texel buffers
	UniformTexelBuffers(Vec<VulkanBufferView>),
}

/// The descriptor set properties
#[derive(Default, Debug, Clone)]
pub struct DescriptorProps {
	/// The descriptor sets
	pub sets: HashMap<u32 /* set */, HashMap<u32 /* binding */, Arc<DescriptorProp>>>,
}
```

### Pipeline

The pipeline wires mesh, texture, uniform buffers, storage buffers, shaders, output images, all together, and defines all of the rendering options.

```rust
let pipeline = ctx.create_pipeline_builder(mesh, draw_shaders, desc_props.clone())?
.set_cull_mode(VkCullModeFlagBits::VK_CULL_MODE_NONE as VkCullModeFlags)
.set_depth_test(false)
.set_depth_write(false)
.build()?;
```

On draw:
```rust
let scene = ctx.begin_scene(0, None)?;
scene.set_viewport_swapchain(0.0, 1.0)?;
scene.set_scissor_swapchain()?;
scene.begin_renderpass(Vec4::new(0.0, 0.0, 0.2, 1.0), 1.0, 0)?;
pipeline.draw(scene.get_cmdbuf())?;
scene.end_renderpass()?;
scene.finish();
```

## Example code

```rust
use vkobject_rs::prelude::*;
use std::{
	collections::HashMap,
	path::PathBuf,
	slice::from_raw_parts_mut,
	sync::{Arc, Mutex},
};

#[derive(Debug)]
pub struct AppInstance {
	pub ctx: VulkanContext,
	pub window: PWindow,
	pub events: GlfwReceiver<(f64, WindowEvent)>,
	pub glfw: Glfw,
	pub num_frames: u64,
}

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

impl AppInstance {
	pub fn new(width: u32, height: u32, title: &str, window_mode: glfw::WindowMode) -> Result<Self, VulkanError> {
		let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
		glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
		let (mut window, events) = glfw.create_window(width, height, title, window_mode).expect("Failed to create GLFW window.");
		window.set_key_polling(true);
		let ctx = create_vulkan_context(&window, PresentInterval::VSync, 1, false)?;
		Ok(Self {
			glfw,
			window,
			events,
			num_frames: 0,
			ctx,
		})
	}

	pub fn run(&mut self,
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
		}
		Ok(())
	}
}

unsafe impl Send for AppInstance {}
unsafe impl Sync for AppInstance {}

fn main() {
	let mut inst = Box::new(AppInstance::new(1024, 768, "GLFW Window", glfw::WindowMode::Windowed).unwrap());

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
			let vertices = BufferWithType::new(device.clone(), &vertices_data, pool_in_use.cmdbuf, VkBufferUsageFlagBits::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as VkBufferUsageFlags)?;
			let mesh = Arc::new(Mutex::new(GenericMeshWithMaterial::new(Box::new(Mesh::new(VkPrimitiveTopology::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, vertices, buffer_unused(), buffer_unused(), buffer_unused())), None)));
			mesh.lock().unwrap().mesh.flush(pool_in_use.cmdbuf)?;
			drop(pool_in_use);
			ctx.cmdpools[0].wait_for_submit(u64::MAX)?;
			mesh.lock().unwrap().mesh.discard_staging_buffers();
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

	let resources = Resources::new(&mut inst.ctx).unwrap();

	inst.run(Some(TEST_TIME),
	|ctx: &mut VulkanContext, run_time: f64| -> Result<(), VulkanError> {
		resources.draw(ctx, run_time)
	}).unwrap();
}
```
