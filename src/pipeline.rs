
use crate::prelude::*;
use std::{
	fmt::Debug,
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

#[derive(Debug, Clone, Copy)]
pub struct Pipeline {
	/// The pipeline
	pipeline: VkPipeline,
}
