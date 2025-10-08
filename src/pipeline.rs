
use crate::prelude::*;
use std::{
	collections::{BTreeMap, HashMap, HashSet, hash_map::Entry},
	fmt::{self, Debug, Formatter},
	ptr::null,
	sync::Arc,
};
use struct_iterable::Iterable;
use shader_analyzer::*;
use rspirv::spirv::*;

/// The trait that the struct of vertices or instances must implement
pub trait VertexType: Copy + Clone + Sized + Default + Debug + Iterable {}
impl<T> VertexType for T where T: Copy + Clone + Sized + Default + Debug + Iterable {}

#[macro_export]
macro_rules! derive_vertex_type {
	($item: item) => {
		#[repr(C)]
		#[derive(Iterable, Default, Debug, Clone, Copy)]
		$item
	};
}

/// The descriptor set layout object
#[derive(Debug)]
pub struct DescriptorSetLayout {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The Vulkan handle of the descriptor set layout
	pub(crate) descriptor_set_layout: VkDescriptorSetLayout,
}

impl DescriptorSetLayout {
	/// Create the `DescriptorSetLayout` by a `VkDescriptorSetLayoutCreateInfo`
	pub fn new(device: Arc<VulkanDevice>, layout_ci: &VkDescriptorSetLayoutCreateInfo) -> Result<Self, VulkanError> {
		let mut descriptor_set_layout = null();
		device.vkcore.vkCreateDescriptorSetLayout(device.get_vk_device(), layout_ci, null(), &mut descriptor_set_layout)?;
		Ok(Self {
			device,
			descriptor_set_layout,
		})
	}
}

impl Drop for DescriptorSetLayout {
	fn drop(&mut self) {
		proceed_run(self.device.vkcore.vkDestroyDescriptorSetLayout(self.device.get_vk_device(), self.descriptor_set_layout, null()));
	}
}

/// Get through an array to know how many dimensions it is, and the final type of it.
/// * This function runs recursively, it could consume stack memory as much as the number of the array dimensions.
pub(crate) fn through_array<'a>(array_info: &'a ArrayType, dimensions: &mut Vec<usize>) -> &'a VariableType {
	dimensions.push(array_info.element_count);
	if let VariableType::Array(sub_array_info) = &array_info.element_type {
		through_array(sub_array_info, dimensions)
	} else {
		&array_info.element_type
	}
}

/// Dig through a multi-dimensional array and get the total size, the element type of the array.
/// * This function calls `through_array()` to dig through the array, which consumes stack memory as much as the number of the array dimensions.
pub(crate) fn dig_array(array_info: &ArrayType) -> (usize, &VariableType) {
	let mut dimensions = Vec::new();
	let var_type = through_array(array_info, &mut dimensions);
	let mut total = 1;
	for dim in dimensions.iter() {
		total *= dim;
	}
	(total, var_type)
}

/// The descriptor set object
pub struct DescriptorSets {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The properties of the shader descriptor sets
	pub desc_props: Arc<DescriptorProps>,

	/// The pool
	pub desc_pool: Arc<DescriptorPool>,

	/// The descriptor set layouts. The key is the set.
	descriptor_set_layouts: BTreeMap<u32 /* set */, DescriptorSetLayout>,

	/// The descriptor sets. The key is the set.
	descriptor_sets: BTreeMap<u32 /* set */, VkDescriptorSet>,

	/// The associated shader
	pub shaders: Arc<DrawShaders>,
}

/// The `WriteDescriptorSets` is to tell the pipeline how are the data get into the pipeline.
/// * e.g. uniform buffers. texture inputs, mesh vertex inputs, etc.
#[derive(Debug, Clone)]
pub struct WriteDescriptorSets {
	pub write_descriptor_sets: Vec<VkWriteDescriptorSet>,
	pub buffer_info: Vec<VkDescriptorBufferInfo>,
	pub image_info: Vec<VkDescriptorImageInfo>,
	pub texel_buffer_views: Vec<VkBufferView>,
}

impl WriteDescriptorSets {
	/// Create a new `WriteDescriptorSets`
	pub(crate) fn new() -> Self {
		Self {
			write_descriptor_sets: Vec::new(),
			buffer_info: Vec::new(),
			image_info: Vec::new(),
			texel_buffer_views: Vec::new(),
		}
	}

	/// Pass through all of the structure tree, use `pass_for_cap = true` to calculate and allocate the needed size of the memory, and use `pass_for_cap = false` to actually generate the structures, passing pointers of structures in the pre-allocated memory.
	fn pass(&mut self, descriptor_sets: &DescriptorSets, shaders: &DrawShaders, pass_for_cap: bool) -> Result<bool, VulkanError> {
		self.buffer_info.clear();
		self.image_info.clear();
		self.texel_buffer_views.clear();
		self.write_descriptor_sets.clear();
		let mut num_buffer_info = 0;
		let mut num_image_info = 0;
		let mut num_texel_buffer_views = 0;
		let mut num_wds = 0;
		let buffer_info_ptr = self.buffer_info.as_ptr();
		let image_info_ptr = self.image_info.as_ptr();
		let texel_buffer_views_ptr = self.texel_buffer_views.as_ptr();
		let mut processed_vars = HashSet::new();
		let desc_props = descriptor_sets.desc_props.clone();
		for (_, shader) in shaders.iter_shaders() {
			for var in shader.get_vars() {
				if let VariableLayout::Descriptor{set, binding, input_attachment_index: iai} = var.layout {
					let var_ident = format!("{set}_{binding}_{iai:?}_{}", var.var_name);
					if processed_vars.contains(&var_ident) {
						continue;
					} else {
						processed_vars.insert(var_ident);
					}
					let mut dimensions = Vec::new();
					let (total_element_count, var_type) = match &var.var_type {
						VariableType::Array(array_info) => {
							let var_type = through_array(array_info, &mut dimensions);
							let mut total = 1;
							for dim in dimensions.iter() {
								total *= dim;
							}
							(total, var_type)
						},
						_ => (1, &var.var_type),
					};
					match var.storage_class {
						StorageClass::Uniform => {
							if pass_for_cap {
								match var_type {
									VariableType::Struct(_) => {
										num_buffer_info += total_element_count;
										num_wds += 1;
									}
									VariableType::Image(_) => {
										num_texel_buffer_views += total_element_count;
										num_wds += 1;
									}
									others => return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of uniform {}: {others:?}", var.var_name))),
								}
							} else {
								match var_type {
									VariableType::Struct(_) => {
										let buffers: Vec<_> = desc_props.get_desc_props_uniform_buffers(set, binding, total_element_count)?.iter().collect();
										if buffers.len() != total_element_count {
											return Err(VulkanError::ShaderInputLengthMismatch(format!("The uniform buffer is `{:?}{}`, need {total_element_count} buffers in total, but {} buffers were given.",
												var.var_type,
												if dimensions.is_empty() {String::new()} else {dimensions.iter().map(|d|format!("[{d}]")).collect::<Vec<_>>().join("")},
												buffers.len()
											)));
										}
										let buffer_info_index = self.buffer_info.len();
										for buffer in buffers.iter() {
											self.buffer_info.push(VkDescriptorBufferInfo {
												buffer: buffer.get_vk_buffer(),
												offset: 0,
												range: buffer.get_size(),
											});
										}
										self.write_descriptor_sets.push(VkWriteDescriptorSet {
											sType: VkStructureType::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
											pNext: null(),
											dstSet: *descriptor_sets.get(&set).unwrap(),
											dstBinding: binding,
											dstArrayElement: 0,
											descriptorCount: total_element_count as u32,
											descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
											pImageInfo: null(),
											pBufferInfo: &self.buffer_info[buffer_info_index],
											pTexelBufferView: null(),
										});
									}
									VariableType::Image(_) => {
										let buffers: Vec<_> = desc_props.get_desc_props_uniform_texel_buffers(set, binding, total_element_count)?.iter().collect();
										if buffers.len() != total_element_count {
											return Err(VulkanError::ShaderInputLengthMismatch(format!("The uniform texel buffer is `{:?}{}`, need {total_element_count} buffers in total, but {} buffers were given.",
												var.var_type,
												if dimensions.is_empty() {String::new()} else {dimensions.iter().map(|d|format!("[{d}]")).collect::<Vec<_>>().join("")},
												buffers.len()
											)));
										}
										let texel_buffer_views_index = self.texel_buffer_views.len();
										for buffer in buffers.iter() {
											self.texel_buffer_views.push(buffer.get_vk_buffer_view());
										}
										self.write_descriptor_sets.push(VkWriteDescriptorSet {
											sType: VkStructureType::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
											pNext: null(),
											dstSet: *descriptor_sets.get(&set).unwrap(),
											dstBinding: binding,
											dstArrayElement: 0,
											descriptorCount: total_element_count as u32,
											descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
											pImageInfo: null(),
											pBufferInfo: null(),
											pTexelBufferView: &self.texel_buffer_views[texel_buffer_views_index],
										});
									}
									others => return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of uniform {}: {others:?}", var.var_name))),
								}
							}
						}
						StorageClass::StorageBuffer => {
							if pass_for_cap {
								match var_type {
									VariableType::Struct(_) => {
										num_buffer_info += total_element_count;
										num_wds += 1;
									}
									VariableType::Image(_) => {
										num_texel_buffer_views += total_element_count;
										num_wds += 1;
									}
									others => return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of storage buffer {}: {others:?}", var.var_name))),
								}
							} else {
								match var_type {
									VariableType::Struct(_) => {
										let buffers: Vec<_> = desc_props.get_desc_props_storage_buffers(set, binding, total_element_count)?.iter().collect();
										if buffers.len() != total_element_count {
											return Err(VulkanError::ShaderInputLengthMismatch(format!("The storage buffer is `{:?}{}`, need {total_element_count} buffers in total, but {} buffers were given.",
												var.var_type,
												if dimensions.is_empty() {String::new()} else {dimensions.iter().map(|d|format!("[{d}]")).collect::<Vec<_>>().join("")},
												buffers.len()
											)));
										}
										let buffer_info_index = self.buffer_info.len();
										for buffer in buffers.iter() {
											self.buffer_info.push(VkDescriptorBufferInfo {
												buffer: buffer.get_vk_buffer(),
												offset: 0,
												range: buffer.get_size(),
											});
										}
										self.write_descriptor_sets.push(VkWriteDescriptorSet {
											sType: VkStructureType::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
											pNext: null(),
											dstSet: *descriptor_sets.get(&set).unwrap(),
											dstBinding: binding,
											dstArrayElement: 0,
											descriptorCount: total_element_count as u32,
											descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
											pImageInfo: null(),
											pBufferInfo: &self.buffer_info[buffer_info_index],
											pTexelBufferView: null(),
										});
									}
									VariableType::Image(_) => {
										let buffers: Vec<_> = desc_props.get_desc_props_storage_texel_buffers(set, binding, total_element_count)?.iter().collect();
										if buffers.len() != total_element_count {
											return Err(VulkanError::ShaderInputLengthMismatch(format!("The storage texel buffer is `{:?}{}`, need {total_element_count} buffers in total, but {} buffers were given.",
												var.var_type,
												if dimensions.is_empty() {String::new()} else {dimensions.iter().map(|d|format!("[{d}]")).collect::<Vec<_>>().join("")},
												buffers.len()
											)));
										}
										let texel_buffer_views_index = self.texel_buffer_views.len();
										for buffer in buffers.iter() {
											self.texel_buffer_views.push(buffer.get_vk_buffer_view());
										}
										self.write_descriptor_sets.push(VkWriteDescriptorSet {
											sType: VkStructureType::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
											pNext: null(),
											dstSet: *descriptor_sets.get(&set).unwrap(),
											dstBinding: binding,
											dstArrayElement: 0,
											descriptorCount: total_element_count as u32,
											descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
											pImageInfo: null(),
											pBufferInfo: null(),
											pTexelBufferView: &self.texel_buffer_views[texel_buffer_views_index],
										});
									}
									others => return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of storage buffer {}: {others:?}", var.var_name))),
								}
							}
						}
						StorageClass::UniformConstant => {
							match var_type {
								VariableType::Literal(literal_type) => {
									if literal_type == "sampler" {
										if pass_for_cap {
											num_image_info += total_element_count;
											num_wds += 1;
										}
										else {
											let samplers: Vec<VkSampler> = desc_props.get_desc_props_samplers(set, binding, total_element_count)?.iter().map(|s|s.get_vk_sampler()).collect();
											let image_info_index = self.image_info.len();
											for sampler in samplers.iter() {
												self.image_info.push(VkDescriptorImageInfo {
													sampler: *sampler,
													imageView: null(),
													imageLayout: VkImageLayout::VK_IMAGE_LAYOUT_UNDEFINED,
												});
											}
											self.write_descriptor_sets.push(VkWriteDescriptorSet {
												sType: VkStructureType::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
												pNext: null(),
												dstSet: *descriptor_sets.get(&set).unwrap(),
												dstBinding: binding,
												dstArrayElement: 0,
												descriptorCount: total_element_count as u32,
												descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_SAMPLER,
												pImageInfo: &self.image_info[image_info_index],
												pBufferInfo: null(),
												pTexelBufferView: null(),
											});
										}
									} else {
										return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of uniform constant input {literal_type}.")));
									}
								}
								VariableType::Image(_) => {
									if pass_for_cap {
										num_image_info += total_element_count;
										num_wds += 1;
									} else {
										let textures: Vec<&TextureForSample> = desc_props.get_desc_props_textures(set, binding, total_element_count)?.iter().collect();
										let image_info_index = self.image_info.len();
										for texture in textures.iter() {
											self.image_info.push(VkDescriptorImageInfo {
												sampler: texture.sampler.get_vk_sampler(),
												imageView: texture.texture.get_vk_image_view(),
												imageLayout: VkImageLayout::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
											});
										}
										self.write_descriptor_sets.push(VkWriteDescriptorSet {
											sType: VkStructureType::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
											pNext: null(),
											dstSet: *descriptor_sets.get(&set).unwrap(),
											dstBinding: binding,
											dstArrayElement: 0,
											descriptorCount: total_element_count as u32,
											descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
											pImageInfo: &self.image_info[image_info_index],
											pBufferInfo: null(),
											pTexelBufferView: null(),
										});
									}
								}
								others => return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of uniform constant {}: {others:?}", var.var_name))),
							}
						}
						// Ignore other storage classes
						_ => {}
					}
				}
			}
		}
		if pass_for_cap {
			self.buffer_info.reserve(num_buffer_info);
			self.image_info.reserve(num_image_info);
			self.texel_buffer_views.reserve(num_texel_buffer_views);
			self.write_descriptor_sets.reserve(num_wds);
		}
		if buffer_info_ptr != self.buffer_info.as_ptr() ||
		   image_info_ptr != self.image_info.as_ptr() ||
		   texel_buffer_views_ptr != self.texel_buffer_views.as_ptr() {
			Ok(false)
		} else {
			Ok(true)
		}
	}

	/// Build inputs to the pipeline
	pub fn build(device: Arc<VulkanDevice>, descriptor_sets: &DescriptorSets, shaders: &DrawShaders) -> Result<(), VulkanError> {
		let mut ret = Self::new();
		ret.pass(descriptor_sets, shaders, true)?;
		assert!(ret.pass(descriptor_sets, shaders, false)?, "The vector pointer changed while pushing data into it, but its capacity should be enough not to trigger the internal memory reallocation. Redesign of the code is needed.");
		if !ret.write_descriptor_sets.is_empty() {
			device.vkcore.vkUpdateDescriptorSets(device.get_vk_device(), ret.write_descriptor_sets.len() as u32, ret.write_descriptor_sets.as_ptr(), 0, null())?;
		}
		Ok(())
	}
}

impl DescriptorSets {
	/// Create the `DescriptorSetLayout` by parsing the shader variable
	pub fn new(device: Arc<VulkanDevice>, desc_pool: Arc<DescriptorPool>, shaders: Arc<DrawShaders>, desc_props: Arc<DescriptorProps>) -> Result<Self, VulkanError> {
		let mut samplers: HashMap<u32 /* set */, HashMap<u32 /* binding */, Vec<VkSampler>>> = HashMap::new();
		let mut layout_bindings: HashMap<u32 /* set */, HashMap<u32 /* binding */, VkDescriptorSetLayoutBinding>> = HashMap::new();
		for (shader_stage, shader) in shaders.iter_shaders() {
			let shader_stage = shader_stage as VkShaderStageFlags;
			for var in shader.get_vars() {
				if let VariableLayout::Descriptor{set, binding, input_attachment_index: _} = var.layout {
					let (total_element_count, var_type) = match &var.var_type {
						VariableType::Array(array_info) => dig_array(array_info),
						others => (1, others),
					};
					match var.storage_class {
						StorageClass::UniformConstant => match var_type {
							VariableType::Literal(literal_type) => {
								if literal_type == "sampler" {
									let samplers = Self::get_samplers_from_map(&mut samplers, set, binding, || Ok(desc_props.get_desc_props_samplers(set, binding, total_element_count)?.iter().map(|s|s.get_vk_sampler()).collect()))?;
									Self::update_desc_set_layout_binding(&mut layout_bindings, set, binding, VkDescriptorSetLayoutBinding {
										binding,
										descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_SAMPLER,
										descriptorCount: total_element_count as u32,
										stageFlags: shader_stage,
										pImmutableSamplers: samplers.as_ptr(),
									})?;
								} else {
									return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of uniform constant {}: {var_type:?}", var.var_name)));
								}
							}
							VariableType::Image(_) => {
								let samplers = Self::get_samplers_from_map(&mut samplers, set, binding, || Ok(desc_props.get_desc_props_textures(set, binding, total_element_count)?.iter().map(|t|t.sampler.get_vk_sampler()).collect()))?;
								Self::update_desc_set_layout_binding(&mut layout_bindings, set, binding, VkDescriptorSetLayoutBinding {
									binding,
									descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
									descriptorCount: total_element_count as u32,
									stageFlags: shader_stage,
									pImmutableSamplers: samplers.as_ptr(),
								})?;
							}
							others => return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of uniform constant {}: {others:?}", var.var_name))),
						}
						StorageClass::Uniform => match var_type {
							VariableType::Struct(_) => {
								Self::update_desc_set_layout_binding(&mut layout_bindings, set, binding, VkDescriptorSetLayoutBinding {
									binding,
									descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
									descriptorCount: total_element_count as u32,
									stageFlags: shader_stage,
									pImmutableSamplers: null(),
								})?;
							}
							VariableType::Image(_) => {
								Self::update_desc_set_layout_binding(&mut layout_bindings, set, binding, VkDescriptorSetLayoutBinding {
									binding,
									descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
									descriptorCount: total_element_count as u32,
									stageFlags: shader_stage,
									pImmutableSamplers: null(),
								})?;
							}
							others => return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of uniform {}: {others:?}", var.var_name))),
						}
						StorageClass::StorageBuffer => match var_type {
							VariableType::Struct(_) => {
								Self::update_desc_set_layout_binding(&mut layout_bindings, set, binding, VkDescriptorSetLayoutBinding {
									binding,
									descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
									descriptorCount: total_element_count as u32,
									stageFlags: shader_stage,
									pImmutableSamplers: null(),
								})?;
							}
							VariableType::Image(_) => {
								Self::update_desc_set_layout_binding(&mut layout_bindings, set, binding, VkDescriptorSetLayoutBinding {
									binding,
									descriptorType: VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
									descriptorCount: total_element_count as u32,
									stageFlags: shader_stage,
									pImmutableSamplers: null(),
								})?;
							}
							others => return Err(VulkanError::ShaderInputTypeUnsupported(format!("Unknown type of storage buffer {}: {others:?}", var.var_name))),
						}
						// Ignore other storage classes
						_ => {}
					}
				}
			}
		}
		let mut bindings_of_set: BTreeMap<u32 /* set */, Vec<VkDescriptorSetLayoutBinding>> = BTreeMap::new();
		for (set, bindings) in layout_bindings.iter() {
			for (_, binding_val) in bindings.iter() {
				let array = if let Some(array) = bindings_of_set.get_mut(set) {
					array
				} else {
					bindings_of_set.insert(*set, Vec::new());
					bindings_of_set.get_mut(set).unwrap()
				};
				array.push(*binding_val);
			}
		}
		let mut descriptor_set_layouts: BTreeMap<u32 /* set */, DescriptorSetLayout> = BTreeMap::new();
		for (set, dslb) in bindings_of_set.iter() {
			let layout_ci = VkDescriptorSetLayoutCreateInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				pNext: null(),
				flags: 0,
				bindingCount: dslb.len() as u32,
				pBindings: dslb.as_ptr(),
			};
			descriptor_set_layouts.insert(*set, DescriptorSetLayout::new(device.clone(), &layout_ci)?);
		}
		let layout_array: Vec<VkDescriptorSetLayout> = descriptor_set_layouts.values().map(|v|v.descriptor_set_layout).collect();
		if layout_array.is_empty() {
			Ok(Self {
				device,
				desc_props,
				desc_pool,
				descriptor_set_layouts,
				descriptor_sets: BTreeMap::new(),
				shaders,
			})
		} else {
			let desc_sets_ai = VkDescriptorSetAllocateInfo {
				sType: VkStructureType::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				pNext: null(),
				descriptorPool: desc_pool.get_vk_pool(),
				descriptorSetCount: layout_array.len() as u32,
				pSetLayouts: layout_array.as_ptr(),
			};
			let mut descriptor_set_array: Vec<VkDescriptorSet> = Vec::with_capacity(layout_array.len());
			device.vkcore.vkAllocateDescriptorSets(device.get_vk_device(), &desc_sets_ai, descriptor_set_array.as_mut_ptr())?;
			unsafe {descriptor_set_array.set_len(layout_array.len())};
			let descriptor_sets: BTreeMap<u32, VkDescriptorSet> = layout_bindings.keys().enumerate().map(|(index, set)|(*set, descriptor_set_array[index])).collect();
			let ret = Self {
				device: device.clone(),
				desc_props,
				desc_pool,
				descriptor_set_layouts,
				descriptor_sets,
				shaders: shaders.clone(),
			};
			WriteDescriptorSets::build(device, &ret, &shaders)?;
			Ok(ret)
		}
	}

	/// Get the `VkDescriptorSetLayout`
	pub fn get_descriptor_sets(&self) -> &BTreeMap<u32, VkDescriptorSet> {
		&self.descriptor_sets
	}

	/// Get the descriptor set layouts
	pub fn get_descriptor_set_layouts(&self) -> &BTreeMap<u32, DescriptorSetLayout> {
		&self.descriptor_set_layouts
	}

	/// Get a `VkDescriptorSet` by a set number
	pub fn get(&self, set_number: &u32) -> Option<&VkDescriptorSet> {
		self.descriptor_sets.get(set_number)
	}

	/// Create or modify items in `HashMap<u32, VkDescriptorSetLayoutBinding>`
	fn update_desc_set_layout_binding(map: &mut HashMap<u32, HashMap<u32, VkDescriptorSetLayoutBinding>>, set: u32, binding: u32, mut item: VkDescriptorSetLayoutBinding) -> Result<(), VulkanError> {
		if let Some(bindings) = map.get_mut(&set) {
			if let Some(existing) = bindings.get_mut(&binding) {
				let prev_stage = vk_shader_stage_flags_to_string(existing.stageFlags);
				let curr_stage = vk_shader_stage_flags_to_string(item.stageFlags);
				if existing.descriptorType != item.descriptorType {
					let prev_type = existing.descriptorType;
					let curr_type = item.descriptorType;
					Err(VulkanError::ShaderInputTypeMismatch(format!("In `layout(set = {set}, binding = {binding})`: descriptor type mismatch: one at `{prev_stage}` is `{prev_type:?}`, another at `{curr_stage}` is `{curr_type:?}`")))
				} else if existing.descriptorCount != item.descriptorCount {
					let prev_count = existing.descriptorCount;
					let curr_count = item.descriptorCount;
					Err(VulkanError::ShaderInputTypeMismatch(format!("In `layout(set = {set}, binding = {binding})`: descriptor count mismatch: one at `{prev_stage}` is `{prev_count}`, another at `{curr_stage}` is `{curr_count}`")))
				} else if existing.pImmutableSamplers != item.pImmutableSamplers {
					let prev_samplers = existing.pImmutableSamplers;
					let curr_samplers = item.pImmutableSamplers;
					Err(VulkanError::ShaderInputTypeMismatch(format!("In `layout(set = {set}, binding = {binding})`: descriptor samplers mismatch: one at `{prev_stage}` is `{prev_samplers:?}`, another at `{curr_stage}` is `{curr_samplers:?}`")))
				} else {
					existing.stageFlags |= item.stageFlags;
					Ok(())
				}
			} else {
				item.binding = binding;
				bindings.insert(binding, item);
				Ok(())
			}
		} else {
			item.binding = binding;
			map.insert(set, [(binding, item)].into_iter().collect());
			Ok(())
		}
	}

	/// Update the sampler map and get the samplers array
	fn get_samplers_from_map(map: &mut HashMap<u32, HashMap<u32, Vec<VkSampler>>>, set: u32, binding: u32, on_create: impl FnOnce() -> Result<Vec<VkSampler>, VulkanError>) -> Result<&[VkSampler], VulkanError> {
		if let Entry::Vacant(e) = map.entry(set) {
			e.insert(HashMap::new());
		}
		let bindings = map.get_mut(&set).unwrap();
		if let Entry::Vacant(e) = bindings.entry(binding) {
			e.insert(on_create()?);
		}
		Ok(bindings.get(&binding).unwrap())
	}
}

impl Clone for DescriptorSets {
	fn clone(&self) -> Self {
		Self::new(self.device.clone(), self.desc_pool.clone(), self.shaders.clone(), self.desc_props.clone()).unwrap()
	}
}

impl Drop for DescriptorSets {
	fn drop(&mut self) {
		let descriptor_set_array: Vec<VkDescriptorSet> = self.descriptor_sets.values().copied().collect();
		proceed_run(self.device.vkcore.vkFreeDescriptorSets(self.device.get_vk_device(), self.desc_pool.get_vk_pool(), descriptor_set_array.len() as u32, descriptor_set_array.as_ptr()));
	}
}

impl Debug for DescriptorSets {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("DescriptorSets")
		.field("desc_props", &self.desc_props)
		.field("desc_pool", &self.desc_pool)
		.field("descriptor_set_layouts", &self.descriptor_set_layouts)
		.field("descriptor_sets", &self.descriptor_sets)
		.field("shaders", &self.shaders)
		.finish()
	}
}

unsafe impl Send for DescriptorSets {}
unsafe impl Sync for DescriptorSets {}

/// Build a pipeline step by step
#[derive(Clone)]
pub struct PipelineBuilder {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The meshes to draw
	pub mesh: Arc<GenericMeshWithMaterial>,

	/// The shaders to use
	pub shaders: Arc<DrawShaders>,

	/// The descriptor sets
	pub descriptor_sets: Arc<DescriptorSets>,

	/// The render target props
	pub renderpass: Arc<VulkanRenderPass>,

	/// The pipeline cache
	pub pipeline_cache: Arc<VulkanPipelineCache>,

	/// The rasterization state create info
	pub rasterization_state_ci: VkPipelineRasterizationStateCreateInfo,

	/// The MSAA state create info
	pub msaa_state_ci: VkPipelineMultisampleStateCreateInfo,

	/// The depth stencil state create info
	pub depth_stenctil_ci: VkPipelineDepthStencilStateCreateInfo,

	/// The color blend state create info
	pub color_blend_state_ci: VkPipelineColorBlendStateCreateInfo,

	/// The color blend attachment states
	pub color_blend_attachment_states: Vec<VkPipelineColorBlendAttachmentState>,

	/// The dynamic states
	pub dynamic_states: HashSet<VkDynamicState>,

	/// The pipeline layout was created by providing descriptor layout there.
	pipeline_layout: VkPipelineLayout,
}

impl PipelineBuilder {
	/// Create the `PipelineBuilder`
	pub fn new(device: Arc<VulkanDevice>, mesh: Arc<GenericMeshWithMaterial>, shaders: Arc<DrawShaders>, desc_pool: Arc<DescriptorPool>, desc_props: Arc<DescriptorProps>, renderpass: Arc<VulkanRenderPass>, pipeline_cache: Arc<VulkanPipelineCache>) -> Result<Self, VulkanError> {
		let descriptor_sets = Arc::new(DescriptorSets::new(device.clone(), desc_pool.clone(), shaders.clone(), desc_props.clone())?);
		let mut desc_set_layouts: Vec<VkDescriptorSetLayout> = Vec::with_capacity(5);
		let mut push_constant_ranges: Vec<VkPushConstantRange> = Vec::with_capacity(5);
		for dsl in descriptor_sets.get_descriptor_set_layouts().values() {
			desc_set_layouts.push(dsl.descriptor_set_layout);
		}
		for (stage, shader) in shaders.iter_shaders() {
			for var in shader.get_vars() {
				if StorageClass::PushConstant != var.storage_class {
					continue;
				}
				match &var.var_type {
					VariableType::Struct(st) => {
						for member in st.members.iter() {
							let size = (((member.size_of()? - 1) / 4 + 1) * 4) as u32;
							push_constant_ranges.push(VkPushConstantRange {
								stageFlags: stage as VkShaderStageFlags,
								offset: member.member_offset,
								size,
							});
						}
					}
					_ => {
						let size = (((var.size_of()? - 1) / 4 + 1) * 4) as u32;
						push_constant_ranges.push(VkPushConstantRange {
							stageFlags: stage as VkShaderStageFlags,
							offset: 0,
							size,
						});
					}
				}
			}
		}
		let rasterization_state_ci = VkPipelineRasterizationStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			depthClampEnable: 0,
			rasterizerDiscardEnable: 0,
			polygonMode: VkPolygonMode::VK_POLYGON_MODE_FILL,
			cullMode: VkCullModeFlagBits::VK_CULL_MODE_BACK_BIT as VkCullModeFlags,
			frontFace: VkFrontFace::VK_FRONT_FACE_COUNTER_CLOCKWISE,
			depthBiasEnable: 0,
			depthBiasConstantFactor: 0.0,
			depthBiasClamp: 0.0,
			depthBiasSlopeFactor: 1.0,
			lineWidth: 1.0,
		};
		let msaa_state_ci = VkPipelineMultisampleStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			rasterizationSamples: VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT,
			sampleShadingEnable: 0,
			minSampleShading: 0.0,
			pSampleMask: null(),
			alphaToCoverageEnable: 0,
			alphaToOneEnable: 0,
		};
		let depth_stenctil_ci = VkPipelineDepthStencilStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			depthTestEnable: 1,
			depthWriteEnable: 1,
			depthCompareOp: VkCompareOp::VK_COMPARE_OP_LESS_OR_EQUAL,
			depthBoundsTestEnable: 0,
			stencilTestEnable: 0,
			front: VkStencilOpState {
				failOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				passOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				depthFailOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				compareOp: VkCompareOp::VK_COMPARE_OP_NEVER,
				compareMask: 0xFFFFFFFF,
				writeMask: 0xFFFFFFFF,
				reference: 0,
			},
			back: VkStencilOpState {
				failOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				passOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				depthFailOp: VkStencilOp::VK_STENCIL_OP_KEEP,
				compareOp: VkCompareOp::VK_COMPARE_OP_NEVER,
				compareMask: 0xFFFFFFFF,
				writeMask: 0xFFFFFFFF,
				reference: 0,
			},
			minDepthBounds: 0.0,
			maxDepthBounds: 0.0,
		};
		let mut color_blend_attachment_states: Vec<VkPipelineColorBlendAttachmentState> = Vec::with_capacity(renderpass.attachments.len());
		for attachment in renderpass.attachments.iter() {
			if !attachment.is_depth_stencil {
				color_blend_attachment_states.push(VkPipelineColorBlendAttachmentState {
					blendEnable: 0,
					srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA,
					dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
					colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
					srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA,
					dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
					alphaBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
					colorWriteMask: VkColorComponentFlagBits::combine(&[
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
						VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
					]),
				});
			}
		}
		let color_blend_state_ci = VkPipelineColorBlendStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			logicOpEnable: 0,
			logicOp: VkLogicOp::VK_LOGIC_OP_COPY,
			attachmentCount: 0,
			pAttachments: null(),
			blendConstants: [0_f32; 4_usize],
		};
		let pipeline_layout_ci = VkPipelineLayoutCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			pNext: null(),
			flags: 0,
			setLayoutCount: desc_set_layouts.len() as u32,
			pSetLayouts: desc_set_layouts.as_ptr(),
			pushConstantRangeCount: push_constant_ranges.len() as u32,
			pPushConstantRanges: push_constant_ranges.as_ptr(),
		};
		let dynamic_states = [
			VkDynamicState::VK_DYNAMIC_STATE_VIEWPORT,
			VkDynamicState::VK_DYNAMIC_STATE_SCISSOR,
		].into_iter().collect();
		let mut pipeline_layout = null();
		device.vkcore.vkCreatePipelineLayout(device.get_vk_device(), &pipeline_layout_ci, null(), &mut pipeline_layout)?;
		Ok(Self {
			device,
			mesh,
			shaders,
			descriptor_sets,
			renderpass,
			pipeline_cache,
			rasterization_state_ci,
			msaa_state_ci,
			depth_stenctil_ci,
			color_blend_state_ci,
			color_blend_attachment_states,
			dynamic_states,
			pipeline_layout,
		})
	}

	/// Set depth clamp enabled/disabled
	pub fn set_depth_clamp_mode(mut self, enabled: bool) -> Self {
		self.rasterization_state_ci.depthClampEnable = if enabled {1} else {0};
		self
	}

	/// Set depth clamp enabled/disabled
	pub fn set_disable_fragment_stage(mut self, disabled: bool) -> Self {
		self.rasterization_state_ci.rasterizerDiscardEnable = if disabled {1} else {0};
		self
	}

	/// Set polygon mode
	pub fn set_polygon_mode(mut self, mode: VkPolygonMode) -> Self {
		self.rasterization_state_ci.polygonMode = mode;
		self
	}

	/// Set cull mode
	pub fn set_cull_mode(mut self, mode: VkCullModeFlags) -> Self {
		self.rasterization_state_ci.cullMode = mode;
		self
	}

	/// Set front face
	pub fn set_front_face(mut self, front_face: VkFrontFace) -> Self {
		self.rasterization_state_ci.frontFace = front_face;
		self
	}

	/// Set enable depth bias mode
	pub fn enable_depth_bias(mut self, constant: f32, slope_scale: f32, clamp: f32) -> Self {
		self.rasterization_state_ci.depthBiasEnable = 1;
		self.rasterization_state_ci.depthBiasConstantFactor = constant;
		self.rasterization_state_ci.depthBiasClamp = clamp;
		self.rasterization_state_ci.depthBiasSlopeFactor= slope_scale;
		self
	}

	/// Set disable depth bias mode
	pub fn disable_depth_bias(mut self) -> Self {
		self.rasterization_state_ci.depthBiasEnable = 0;
		self
	}

	/// Set line width
	pub fn set_line_width(mut self, line_width: f32) -> Self {
		self.rasterization_state_ci.lineWidth = line_width;
		self
	}

	/// Set MSAA sample cound
	pub fn set_msaa_samples(mut self, msaa_samples: VkSampleCountFlagBits) -> Self {
		self.msaa_state_ci.rasterizationSamples = msaa_samples;
		self
	}

	/// Set use dithering to handle MSAA alpha
	pub fn set_msaa_alpha_to_coverage(mut self, enabled: bool) -> Self {
		self.msaa_state_ci.alphaToCoverageEnable = if enabled {1} else {0};
		self
	}

	/// Set MSAA supersampling state
	pub fn set_msaa_super_sampling(mut self, quality: Option<f32>) -> Self {
		if let Some(quality) = quality {
			self.msaa_state_ci.sampleShadingEnable = 1;
			self.msaa_state_ci.minSampleShading = quality;
		} else {
			self.msaa_state_ci.sampleShadingEnable = 0;
		}
		self
	}

	/// Set depth test state
	pub fn set_depth_test(mut self, enabled: bool) -> Self {
		self.depth_stenctil_ci.depthTestEnable = if enabled {1} else {0};
		self
	}

	/// Set depth write state
	pub fn set_depth_write(mut self, enabled: bool) -> Self {
		self.depth_stenctil_ci.depthWriteEnable = if enabled {1} else {0};
		self
	}

	/// Set depth compare mode
	pub fn set_depth_compare_mode(mut self, mode: VkCompareOp) -> Self {
		self.depth_stenctil_ci.depthCompareOp = mode;
		self
	}

	/// Set depth bound test mode
	pub fn set_depth_bound_test_mode(mut self, bounds: Option<(f32, f32)>) -> Self {
		if let Some((min_bound, max_bound)) = bounds {
			self.depth_stenctil_ci.depthBoundsTestEnable = 1;
			self.depth_stenctil_ci.minDepthBounds = min_bound;
			self.depth_stenctil_ci.maxDepthBounds = max_bound;
		} else {
			self.depth_stenctil_ci.depthBoundsTestEnable = 0;
		}
		self
	}

	/// Set stencil test mode
	pub fn set_stencil_test(mut self, enabled: bool) -> Self {
		self.depth_stenctil_ci.stencilTestEnable = if enabled {1} else {0};
		self
	}

	/// Set stencil mode
	pub fn set_stencil_mode(mut self, front_face: VkStencilOpState, back_face: VkStencilOpState) -> Self {
		self.depth_stenctil_ci.front = front_face;
		self.depth_stenctil_ci.back = back_face;
		self
	}

	/// Return a color blend mode for normal alpha blend
	pub fn normal_blend_mode() -> VkPipelineColorBlendAttachmentState {
		VkPipelineColorBlendAttachmentState {
			blendEnable: 1,
			srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA,
			dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA,
			dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			alphaBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			colorWriteMask: VkColorComponentFlagBits::combine(&[
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
			]),
		}
	}

	/// Return a color blend mode for additive blend
	pub fn additive_blend_mode() -> VkPipelineColorBlendAttachmentState {
		VkPipelineColorBlendAttachmentState {
			blendEnable: 1,
			srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE,
			dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE,
			colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE,
			dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ONE,
			alphaBlendOp: VkBlendOp::VK_BLEND_OP_MAX,
			colorWriteMask: VkColorComponentFlagBits::combine(&[
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
			]),
		}
	}

	/// Return a color blend mode for no blending
	pub fn disabled_blend_mode() -> VkPipelineColorBlendAttachmentState {
		VkPipelineColorBlendAttachmentState {
			blendEnable: 0,
			srcColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			dstColorBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			colorBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			srcAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			dstAlphaBlendFactor: VkBlendFactor::VK_BLEND_FACTOR_ZERO,
			alphaBlendOp: VkBlendOp::VK_BLEND_OP_ADD,
			colorWriteMask: VkColorComponentFlagBits::combine(&[
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_R_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_G_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_B_BIT,
				VkColorComponentFlagBits::VK_COLOR_COMPONENT_A_BIT,
			]),
		}
	}

	/// Set alpha blend mode
	pub fn set_color_blend_mode(mut self, attachment_index: usize, color_blend_mode: VkPipelineColorBlendAttachmentState) -> Self {
		self.color_blend_attachment_states[attachment_index] = color_blend_mode;
		self
	}

	/// Add a dynamic state
	pub fn add_dynamic_state(mut self, dynamic_state: VkDynamicState) -> Self {
		self.dynamic_states.insert(dynamic_state);
		self
	}

	/// Remove a dynamic state
	pub fn remove_dynamic_state(mut self, dynamic_state: VkDynamicState) -> Self {
		self.dynamic_states.remove(&dynamic_state);
		self
	}

	/// Generate the pipeline
	pub fn build(self) -> Result<Pipeline, VulkanError> {
		Pipeline::new(self)
	}
}

impl Debug for PipelineBuilder {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("PipelineBuilder")
		.field("mesh", &self.mesh)
		.field("shaders", &self.shaders)
		.field("descriptor_sets", &self.descriptor_sets)
		.field("renderpass", &self.renderpass)
		.field("pipeline_cache", &self.pipeline_cache)
		.field("rasterization_state_ci", &self.rasterization_state_ci)
		.field("msaa_state_ci", &self.msaa_state_ci)
		.field("depth_stenctil_ci", &self.depth_stenctil_ci)
		.field("color_blend_state_ci", &self.color_blend_state_ci)
		.field("color_blend_attachment_states", &self.color_blend_attachment_states)
		.field("dynamic_states", &self.dynamic_states)
		.field("pipeline_layout", &self.pipeline_layout)
		.finish()
	}
}

impl Drop for PipelineBuilder {
	fn drop(&mut self) {
		if !self.pipeline_layout.is_null() {
			proceed_run(self.device.vkcore.vkDestroyPipelineLayout(self.device.get_vk_device(), self.pipeline_layout, null()))
		}
	}
}

/// The core thing of Vulkan: the pipeline. This thing manages the inputs to the shaders, outputs of the shaders, and drawing states, all aspects, everything of a rendering behavior.
/// * use `PipelineBuilder` to build a pipeline, and draw with it!
/// * Note: building this thing could be very slow, do not build it on every frame, and only use it once every time and discard it.
pub struct Pipeline {
	/// The associated device
	pub device: Arc<VulkanDevice>,

	/// The meshes to draw
	pub mesh: Arc<GenericMeshWithMaterial>,

	/// The shaders to use
	pub shaders: Arc<DrawShaders>,

	/// The render pass
	pub renderpass: Arc<VulkanRenderPass>,

	/// The pipeline cache
	pub pipeline_cache: Arc<VulkanPipelineCache>,

	/// The descriptor sets
	pub descriptor_sets: Arc<DescriptorSets>,

	/// The descriptor sets to be binded, this is just an accelerator
	descriptor_sets_to_bind: BTreeMap<u32, Vec<VkDescriptorSet>>,

	/// The pipeline layout was created by providing descriptor layout there.
	pipeline_layout: VkPipelineLayout,

	/// The pipeline
	pipeline: VkPipeline,
}

struct MemberInfo<'a> {
	name: &'a str,
	type_name: &'static str,
	row_format: VkFormat,
	num_rows: u32,
	offset: u32,
	size: usize,
}

impl Pipeline {
	/// Create the `Pipeline`
	pub fn new(mut builder: PipelineBuilder) -> Result<Self, VulkanError> {
		let device = builder.device.clone();
		let mesh = builder.mesh.clone();
		let shaders = builder.shaders.clone();
		let renderpass = builder.renderpass.clone();
		let pipeline_cache = builder.pipeline_cache.clone();
		let descriptor_sets = builder.descriptor_sets.clone();
		let pipeline_layout = builder.pipeline_layout;
		builder.pipeline_layout = null();
		let shader_stages: Vec<VkPipelineShaderStageCreateInfo> = shaders.iter_shaders().map(|(stage, shader)| VkPipelineShaderStageCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			stage,
			module: shader.get_vk_shader(),
			pName: shader.get_entry_point().as_ptr(),
			pSpecializationInfo: null(),
		}).collect();
		let type_id_to_info = TypeInfo::get_map_of_type_id_to_info();
		let mut mesh_vertex_inputs: HashMap<String, MemberInfo> = HashMap::new();
		let mut mesh_instance_inputs: HashMap<String, MemberInfo> = HashMap::new();
		let vertex_stride = mesh.geometry.get_vertex_stride();
		let instance_stride = mesh.geometry.get_instance_stride();
		let topology = mesh.geometry.get_primitive_type();
		let mut cur_vertex_offset = 0;
		for (name, var) in mesh.geometry.iter_vertex_buffer_struct_members() {
			if let Some(info) = type_id_to_info.get(&var.type_id()) {
				mesh_vertex_inputs.insert(name.to_string(), MemberInfo {
					name,
					type_name: info.type_name,
					row_format: info.row_format,
					num_rows: info.num_rows,
					offset: cur_vertex_offset,
					size: info.size,
				});
				cur_vertex_offset += info.size as u32;
			} else {
				panic!("Unknown member {:?} of the vertex struct: `{:?}`", var, var.type_id());
			}
		}
		if let Some(instance_member_iter) = mesh.geometry.iter_instance_buffer_struct_members() {
			let mut cur_instance_offset = 0;
			for (name, var) in instance_member_iter {
				if let Some(info) = type_id_to_info.get(&var.type_id()) {
					mesh_instance_inputs.insert(name.to_string(), MemberInfo {
						name,
						type_name: info.type_name,
						row_format: info.row_format,
						num_rows: info.num_rows,
						offset: cur_instance_offset,
						size: info.size,
					});
					cur_instance_offset += info.size as u32;
				} else {
					panic!("Unknown member {:?} of the instance struct: `{:?}`", var, var.type_id());
				}
			}
		}
		let mut vertex_input_bindings: Vec<VkVertexInputBindingDescription> = Vec::with_capacity(2);
		vertex_input_bindings.push(VkVertexInputBindingDescription {
			binding: 0,
			stride: vertex_stride as u32,
			inputRate: VkVertexInputRate::VK_VERTEX_INPUT_RATE_VERTEX,
		});
		if !mesh_instance_inputs.is_empty() {
			vertex_input_bindings.push(VkVertexInputBindingDescription {
				binding: 1,
				stride: instance_stride as u32,
				inputRate: VkVertexInputRate::VK_VERTEX_INPUT_RATE_INSTANCE,
			});
		}
		let mut vertex_attrib_bindings: Vec<VkVertexInputAttributeDescription> = Vec::with_capacity(mesh_vertex_inputs.len() + mesh_instance_inputs.len());
		for var in shaders.vertex_shader.get_vars() {
			if let VariableLayout::Location(location) = var.layout {
				if let Some(member_info) = mesh_vertex_inputs.get(&var.var_name) {
					let row_stride = member_info.size as u32 / member_info.num_rows;
					for row in 0..member_info.num_rows {
						vertex_attrib_bindings.push(VkVertexInputAttributeDescription {
							location: location + row,
							binding: 0,
							format: member_info.row_format,
							offset: member_info.offset + row * row_stride,
						});
					}
				} else if let Some(member_info) = mesh_instance_inputs.get(&var.var_name) {
					let row_stride = member_info.size as u32 / member_info.num_rows;
					for row in 0..member_info.num_rows {
						vertex_attrib_bindings.push(VkVertexInputAttributeDescription {
							location: location + row,
							binding: 1,
							format: member_info.row_format,
							offset: member_info.offset + row * row_stride,
						});
					}
				}
			}
		}
		let vertex_input_state_ci = VkPipelineVertexInputStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			vertexBindingDescriptionCount: vertex_input_bindings.len() as u32,
			pVertexBindingDescriptions: vertex_input_bindings.as_ptr(),
			vertexAttributeDescriptionCount: vertex_attrib_bindings.len() as u32,
			pVertexAttributeDescriptions: vertex_attrib_bindings.as_ptr(),
		};
		let input_assembly_state_ci = VkPipelineInputAssemblyStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			topology,
			primitiveRestartEnable: 0,
		};
		let viewport_state_ci = VkPipelineViewportStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			viewportCount: 1,
			pViewports: null(),
			scissorCount: 1,
			pScissors: null(),
		};
		builder.color_blend_state_ci.attachmentCount = builder.color_blend_attachment_states.len() as u32;
		builder.color_blend_state_ci.pAttachments = builder.color_blend_attachment_states.as_ptr();
		let dynamic_states: Vec<VkDynamicState> = builder.dynamic_states.clone().into_iter().collect();
		let dynamic_state_ci = VkPipelineDynamicStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			dynamicStateCount: dynamic_states.len() as u32,
			pDynamicStates: dynamic_states.as_ptr(),
		};
		let tessellation_state_ci = VkPipelineTessellationStateCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			patchControlPoints: if let Some(tcs) = &shaders.tessellation_control_shader {tcs.get_tessellation_output_vertices().unwrap_or(0)} else {0},
		};
		let pipeline_ci = VkGraphicsPipelineCreateInfo {
			sType: VkStructureType::VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			pNext: null(),
			flags: 0,
			stageCount: shader_stages.len() as u32,
			pStages: shader_stages.as_ptr(),
			pVertexInputState: &vertex_input_state_ci,
			pInputAssemblyState: &input_assembly_state_ci,
			pTessellationState: if tessellation_state_ci.patchControlPoints == 0 {null()} else {&tessellation_state_ci},
			pViewportState: &viewport_state_ci,
			pRasterizationState: &builder.rasterization_state_ci,
			pMultisampleState: &builder.msaa_state_ci,
			pDepthStencilState: &builder.depth_stenctil_ci,
			pColorBlendState: &builder.color_blend_state_ci,
			pDynamicState: &dynamic_state_ci,
			layout: pipeline_layout,
			renderPass: renderpass.get_vk_renderpass(),
			subpass: 0,
			basePipelineHandle: null(),
			basePipelineIndex: 0,
		};
		let mut pipeline = null();
		device.vkcore.vkCreateGraphicsPipelines(device.get_vk_device(), pipeline_cache.get_vk_pipeline_cache(), 1, &pipeline_ci, null(), &mut pipeline)?;
		let mut descriptor_sets_to_bind: BTreeMap<u32, Vec<VkDescriptorSet>> = BTreeMap::new();
		let descriptor_sets_map = descriptor_sets.get_descriptor_sets();
		if !descriptor_sets_map.is_empty() {
			let first_set = *descriptor_sets_map.keys().next().unwrap();
			let last_set = *descriptor_sets_map.last_key_value().unwrap().0;
			let mut prev_set = None;
			for i in first_set..=last_set {
				if let Some(set) = descriptor_sets_map.get(&i) {
					if let Some(first_set) = &prev_set {
						descriptor_sets_to_bind.get_mut(first_set).unwrap().push(*set);
					} else {
						prev_set = Some(i);
						descriptor_sets_to_bind.insert(i, vec![*set]);
					}
				} else {
					prev_set = None;
				}
			}
		}
		Ok(Self {
			device,
			mesh,
			shaders,
			renderpass,
			pipeline_cache,
			descriptor_sets,
			descriptor_sets_to_bind,
			pipeline_layout,
			pipeline,
		})
	}

	/// Get the descriptor set layouts
	pub(crate) fn get_vk_pipeline(&self) -> VkPipeline {
		self.pipeline
	}

	/// Invoke `vkCmdBindDescriptorSets`
	fn bind_descriptor_sets(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		for (first_set, sets) in self.descriptor_sets_to_bind.iter() {
			self.device.vkcore.vkCmdBindDescriptorSets(cmdbuf, VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout, *first_set, sets.len() as u32, sets.as_ptr(), 0, null())?;
		}
		Ok(())
	}

	/// Prepare data to draw
	pub fn prepare_data(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		self.mesh.geometry.flush(cmdbuf)?;
		Ok(())
	}

	/// Queue draw command
	pub fn draw(&self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		let vkcore = &self.device.vkcore;
		self.bind_descriptor_sets(cmdbuf)?;
		vkcore.vkCmdBindPipeline(cmdbuf, VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline)?;
		let vertex_buffer = self.mesh.geometry.get_vk_vertex_buffer();
		let index_buffer = self.mesh.geometry.get_vk_index_buffer();
		let instance_buffer = self.mesh.geometry.get_vk_instance_buffer();
		let command_buffer = self.mesh.geometry.get_vk_command_buffer();
		let vertex_count = self.mesh.geometry.get_vertex_count() as u32;
		let index_count = self.mesh.geometry.get_index_count() as u32;
		let instance_count = self.mesh.geometry.get_instance_count() as u32;
		let command_count = self.mesh.geometry.get_command_count() as u32;
		let index_type = self.mesh.geometry.get_index_type().unwrap_or(VkIndexType::VK_INDEX_TYPE_UINT16);
		let command_stride = self.mesh.geometry.get_command_stride() as u32;
		if let Some(index_buffer) = index_buffer {
			vkcore.vkCmdBindIndexBuffer(cmdbuf, index_buffer, 0, index_type)?;
		}
		if let Some(instance_buffer) = instance_buffer {
			let vertex_buffers = [vertex_buffer, instance_buffer];
			let offsets = [0, 0];
			vkcore.vkCmdBindVertexBuffers(cmdbuf, 0, vertex_buffers.len() as u32, vertex_buffers.as_ptr(), offsets.as_ptr())?;
		} else {
			let vertex_buffers = [vertex_buffer];
			let offsets = [0];
			vkcore.vkCmdBindVertexBuffers(cmdbuf, 0, vertex_buffers.len() as u32, vertex_buffers.as_ptr(), offsets.as_ptr())?;
		}
		match (index_buffer, command_buffer) {
			(None, None) => vkcore.vkCmdDraw(cmdbuf, vertex_count, instance_count, 0, 0)?,
			(Some(_), None) => vkcore.vkCmdDrawIndexed(cmdbuf, index_count, instance_count, 0, 0, 0)?,
			(None, Some(buffer)) => vkcore.vkCmdDrawIndirect(cmdbuf, buffer, 0, command_count, command_stride)?,
			(Some(_), Some(buffer)) => vkcore.vkCmdDrawIndexedIndirect(cmdbuf, buffer, 0, command_count, command_stride)?,
		}
		Ok(())
	}
}

impl Debug for Pipeline {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("Pipeline")
		.field("mesh", &self.mesh)
		.field("shaders", &self.shaders)
		.field("renderpass", &self.renderpass)
		.field("pipeline_cache", &self.pipeline_cache)
		.field("descriptor_sets", &self.descriptor_sets)
		.field("descriptor_sets_to_bind", &self.descriptor_sets_to_bind)
		.field("pipeline_layout", &self.pipeline_layout)
		.field("pipeline", &self.pipeline)
		.finish()
	}
}

impl Drop for Pipeline {
	fn drop(&mut self) {
		proceed_run(self.device.wait_idle());
		proceed_run(self.device.vkcore.vkDestroyPipelineLayout(self.device.get_vk_device(), self.pipeline_layout, null()));
		proceed_run(self.device.vkcore.vkDestroyPipeline(self.device.get_vk_device(), self.pipeline, null()));
	}
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}
