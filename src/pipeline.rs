
use crate::prelude::*;
use std::{
	fmt::Debug,
	ptr::null,
	sync::{Arc, Mutex},
};
use struct_iterable::Iterable;

/// The trait that the struct of vertices or instances must implement
pub trait VertexType: Copy + Clone + Sized + Default + Debug + Iterable {}
impl<T> VertexType for T where T: Copy + Clone + Sized + Default + Debug + Iterable {}

#[macro_export]
macro_rules! derive_vertex_type {
	($item: item) => {
		#[derive(Iterable, Default, Debug, Clone, Copy)]
		$item
	};
}

/// The shaders to use
#[derive(Debug, Clone)]
pub struct DrawShaders {
	/// The vertex shader cannot be absent
	vertex_shader: Arc<VulkanShader>,

	/// The optional tessellation control shader
	tessellation_control_shader: Option<Arc<VulkanShader>>,

	/// The optional tessellation evaluation shader
	tessellation_evaluation_shader: Option<Arc<VulkanShader>>,

	/// The optional geometry shader
	geometry_shader: Option<Arc<VulkanShader>>,

	/// The fragment shader cannot be absent
	fragment_shader: Arc<VulkanShader>,
}

impl DrawShaders {
	/// Create the `DrawShaders`
	pub fn new(
		vertex_shader: Arc<VulkanShader>,
		tessellation_control_shader: Option<Arc<VulkanShader>>,
		tessellation_evaluation_shader: Option<Arc<VulkanShader>>,
		geometry_shader: Option<Arc<VulkanShader>>,
		fragment_shader: Arc<VulkanShader>) -> Self {
		Self {
			vertex_shader,
			tessellation_control_shader,
			tessellation_evaluation_shader,
			geometry_shader,
			fragment_shader,
		}
	}

	/// Create an iterator that iterates through all of the shaders variables
	pub fn iter_vars(&self) -> impl Iterator<Item = &Arc<ShaderVariable>> {
		self.vertex_shader.get_vars().iter().chain(
		if let Some(geometry_shader) = &self.geometry_shader {
			geometry_shader.get_vars().iter()
		} else {
			[].iter()
		}).chain(
			self.fragment_shader.get_vars().iter()
		)
	}
}

unsafe impl Send for DrawShaders {}
unsafe impl Sync for DrawShaders {}

#[derive(Debug)]
pub struct Pipeline {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The pipeline
	pipeline: VkPipeline,

	/// The descriptor set layouts of the shaders
	descriptor_set_layouts: Vec<DescriptorSetLayout>,

	/// The mesh to draw
	pub mesh: Arc<Mutex<GenericMeshWithMaterial>>,

	/// The shaders to use
	pub shaders: Arc<DrawShaders>,
}

impl Pipeline {
	/// Create the `Pipeline`
	pub fn new(mesh: Arc<Mutex<GenericMeshWithMaterial>>, shaders: Arc<DrawShaders>) -> Result<Self, VulkanError> {
		Ok(Self {
			pipeline: null(),
			mesh,
			shaders,
		})
	}
}
