# Vulkan Object Wrapper：另一个 Rust 渲染引擎实现

## 语言｜language

Readme-CN.md | [Chinglish](Readme.md)

## 用法

### 添加到项目
```cmd
cargo add vkobject-rs
```

## 渲染策略

1. 创建 `VulkanContext`
- 如果你想在 GLFW 中使用 **vkobject-rs**，最简单的方法是在你的 GLFW 窗口中使用 `create_vulkan_context()`。

2. 构建用于绘制的管线。
- 管线是一个对象，它将所有缓冲区中的所有数据收集到着色器中，运行着色器，然后将输出连接到渲染目标附件。
- 要构建管线，你必须创建以下内容：
	1. 网格。网格描述你要渲染的多边形。
	2. 着色器。着色器定义如何处理多边形以生成像素，并将像素写入渲染目标中。
	3. 着色器的输入数据。uniform buffer、storage buffer、push constants、textures, or texel buffers 都可以实现这一点。

3. 使用管道进行绘制。
- 请参阅**示例代码**部分中 `pub fn draw()` 的实现。

4. 当对象超出作用域时，资源清理会自动完成，靠的是 Rust 语言的 RAII 规则。

## 主要使用的对象

### 缓冲区：用于保存多边形、绘制实例、绘制命令和着色器存储空间

此 crate 中包含以下几种类型的缓冲区：
- `BufferWithType<T>`：`Buffer` 的包装器，主要用于存储不经常修改的数据，因此其暂存缓冲区可以丢弃。
- `BufferVec<T>`：`Buffer` 的包装器，提供类似于 `Vec<T>` 的接口，调用 `flush()` 可以将数据通过命令缓冲区上传到 GPU。
	- `flush()` 只会将修改后的数据上传到 GPU。数据更新是增量式的，从而最大限度地减少了 CPU-GPU 数据传输的带宽占用。
	- 此缓冲区的暂存缓冲区不可丢弃。
- `UniformBuffer<T>`：`Buffer` 的包装器，其 CPU 端的数据可以解引用为结构体，你可以自由修改结构体成员。
	- `flush()` 会将整个结构上传到 GPU 端。
	- 此缓冲区通常用于着色器输入。
- `GenericUniformBuffer`：`UniformBuffer<T>` 的包装器，用于清除泛型类型 `<T>`。
- `StorageBuffer<T>`：Buffer 的包装器，用法与 `UniformBuffer<T>` 相同，只是它用于着色器的 storage buffer 输入。
	- 着色器可以自由修改 storage buffer，但不能修改 uniform buffer。
- `GenericStorageBuffer`：`StorageBuffer<T>` 的包装器，用于清除泛型类型 `<T>`。
- `StagingBuffer`：对 CPU 完全透明的暂存缓冲区，你可以获取其数据指针并修改数据，以便将其上传到 GPU。
	- 安全性：你应该知道如何正确操作原始指针。
- `Buffer`：对 GPU 透明的缓冲区，它拥有自己的暂存缓冲区，用于将数据上传到 GPU。
	- 将数据传输到暂存缓冲区，然后调用 `upload_staging_buffer()` 将上传命令加入命令缓冲区。
	- 上传命令执行后，数据将传输到 GPU，然后调用 `discard_staging_buffer()` 以节省一些系统内存（如果需要）。
	- 这玩意儿是底层的东西，不是给你直接用的。
- `VulkanBuffer`：最底层的缓冲区包装对象，`Buffer` 和 `StagingBuffer` 都是使用此对象实现的。
	- 这玩意儿是最底层的东西，不是给你直接用的。

### 网格：用于保存多边形、绘制实例和绘制命令

网格有 4 个缓冲区。根据用途，它们分别是：
- 顶点缓冲区
- 索引缓冲区（可选）
- 实例缓冲区（可选）
- 间接绘制命令缓冲区（可选）

网格的缓冲区有两种类型：
- 对于静态绘制，可以使用 `BufferWithType<T>`。
	- 缓冲区中的数据一旦初始化，便不会更改。
- 对于动态更新，可以使用 `BufferVec<T>`。
	- 你可以像 `Vec<T>` 一样频繁修改其数据，然后调用 `flush()` 将更改应用于 GPU 缓冲区。

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

### 纹理

`VulkanTexture` 是用于使用纹理的包装器。

### DescriptorProps

描述符属性用于着色器输入；它们定义哪些描述符集和绑定具有统一的缓冲区、纹理、采样器等。
- 着色器输入被设置为 `Vec<T>`，因为这有助于为着色器的数组类型输入提供数据。
- 对于单变量输入，只需提供数组的一个元素即可。

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

### 着色器

着色器对象带有编译功能，可以将 GLSL 或 HLSL 代码编译为 SPIR-V 中间语言。此外，它们也可以从二进制文件（而非源代码）加载。

```rust
let draw_shaders = Arc::new(DrawShaders::new(
	Arc::new(VulkanShader::new_from_source_file_or_cache(device.clone(), ShaderSourcePath::VertexShader(PathBuf::from("shaders/test.vsh")), false, "main", OptimizationLevel::Performance, false)?),
	None,
	None,
	None,
	Arc::new(VulkanShader::new_from_source_file_or_cache(device.clone(), ShaderSourcePath::FragmentShader(PathBuf::from("shaders/test.fsh")), false, "main", OptimizationLevel::Performance, false)?),
));
```

### 管线

管线将网格、纹理、统一缓冲区、存储缓冲区、着色器、输出图像等连接在一起，并定义所有渲染选项。

```rust
let pipeline = ctx.create_pipeline_builder(mesh, draw_shaders, desc_props.clone())?
.set_cull_mode(VkCullModeFlagBits::VK_CULL_MODE_NONE as VkCullModeFlags)
.set_depth_test(false)
.set_depth_write(false)
.build()?;
```

绘制时：
```rust
let scene = ctx.begin_scene(0, None)?;
scene.set_viewport_swapchain(0.0, 1.0)?;
scene.set_scissor_swapchain()?;
scene.begin_renderpass(Vec4::new(0.0, 0.0, 0.2, 1.0), 1.0, 0)?;
pipeline.draw(scene.get_cmdbuf())?;
scene.end_renderpass()?;
scene.finish();
```

## 示例代码

```rust
use glfw::*;
use crate::prelude::*;
use std::{
	collections::HashMap,
	ffi::CStr,
	path::PathBuf,
	slice::from_raw_parts_mut,
	sync::{
		Arc,
		Mutex,
		RwLock,
		atomic::{
			AtomicBool,
			Ordering,
		}
	},
	thread,
	time::Duration,
};

const TEST_TIME: f64 = 10.0;

#[derive(Debug)]
pub struct AppInstance {
	pub ctx: Arc<RwLock<VulkanContext>>,
	pub window: PWindow,
	pub events: GlfwReceiver<(f64, WindowEvent)>,
	pub glfw: Glfw,
}

impl AppInstance {
	pub fn new(width: u32, height: u32, title: &str, window_mode: glfw::WindowMode) -> Result<Self, VulkanError> {
		static GLFW_LOCK: Mutex<u32> = Mutex::new(0);
		let glfw_lock = GLFW_LOCK.lock().unwrap();
		let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
		glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
		let (mut window, events) = glfw.create_window(width, height, title, window_mode).expect("Failed to create GLFW window.");
		drop(glfw_lock);
		window.set_key_polling(true);
		let device_requirement = DeviceRequirement {
			can_graphics: true,
			can_compute: false,
			name_subtring: "",
		};
		let ctx = Arc::new(RwLock::new(create_vulkan_context(&window, device_requirement, PresentInterval::VSync, 1, false)?));
		let ctx_lock = ctx.read().unwrap();
		for gpu in VulkanGpuInfo::get_gpu_info(&ctx_lock.vkcore)?.iter() {
			println!("Found GPU: {}", unsafe{CStr::from_ptr(gpu.properties.deviceName.as_ptr())}.to_str().unwrap());
		}
		println!("Chosen GPU name: {}", unsafe{CStr::from_ptr(ctx_lock.device.get_gpu().properties.deviceName.as_ptr())}.to_str().unwrap());
		println!("Chosen GPU type: {:?}", ctx_lock.device.get_gpu().properties.deviceType);
		drop(ctx_lock);
		Ok(Self {
			glfw,
			window,
			events,
			ctx,
		})
	}

	pub fn get_time(&self) -> f64 {
		glfw_get_time()
	}

	pub fn set_time(&self, time: f64) {
		glfw_set_time(time)
	}

	pub fn run(&mut self,
		test_time: Option<f64>,
		mut on_render: impl FnMut(&mut VulkanContext, f64) -> Result<(), VulkanError> + Send + 'static
	) -> Result<(), VulkanError> {
		let exit_flag = Arc::new(AtomicBool::new(false));
		let exit_flag_cloned = exit_flag.clone();
		let start_time = self.glfw.get_time();
		let ctx = self.ctx.clone();
		let renderer_thread = thread::spawn(move || {
			let mut num_frames = 0;
			let mut time_in_sec: u64 = 0;
			let mut num_frames_prev: u64 = 0;
			while !exit_flag_cloned.load(Ordering::Relaxed) {
				let cur_frame_time = glfw_get_time();
				let run_time = cur_frame_time - start_time;
				on_render(&mut ctx.write().unwrap(), run_time).unwrap();
				num_frames += 1;
				let new_time_in_sec = run_time.floor() as u64;
				if new_time_in_sec > time_in_sec {
					let fps = num_frames - num_frames_prev;
					println!("FPS: {fps}\tat {new_time_in_sec}s");
					time_in_sec = new_time_in_sec;
					num_frames_prev = num_frames;
				}
			}
		});
		while !self.window.should_close() {
			let run_time = glfw_get_time() - start_time;
			thread::sleep(Duration::from_millis(1));
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
		exit_flag.store(true, Ordering::Relaxed);
		renderer_thread.join().unwrap();
		println!("End of the test");
		Ok(())
	}
}

unsafe impl Send for AppInstance {}
unsafe impl Sync for AppInstance {}

fn main() {
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
			let desc_props = Arc::new(DescriptorProps::default());
			desc_props.new_uniform_buffer(0, 0, uniform_input.clone());
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
			let pipeline = ctx.create_pipeline_builder(mesh, draw_shaders, desc_props)?
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

	let mut inst = Box::new(AppInstance::new(1024, 768, "Vulkan test", glfw::WindowMode::Windowed).unwrap());
	let resources = Resources::new(&mut inst.ctx.write().unwrap()).unwrap();
	inst.run(Some(TEST_TIME),
	move |ctx: &mut VulkanContext, run_time: f64| -> Result<(), VulkanError> {
		resources.draw(ctx, run_time)
	}).unwrap();
}
```
