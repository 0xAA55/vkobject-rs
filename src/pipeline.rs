
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

#[derive(Debug, Clone)]
pub struct Pipeline {
	/// The pipeline
	pipeline: VkPipeline,

	/// The mesh to draw
	mesh: Arc<Mutex<GenericMeshWithMaterial>>,

	/// The shaders to use
	shaders: Arc<DrawShaders>,
}

impl DrawShaders {
	/// Create the `DrawShaders`
	pub fn new(vertex_shader: Arc<VulkanShader>, geometry_shader: Option<Arc<VulkanShader>>, fragment_shader: Arc<VulkanShader>) -> Self {
		Self {
			vertex_shader,
			geometry_shader,
			fragment_shader,
		}
	}
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
