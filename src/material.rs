
use crate::prelude::*;
use std::{
	collections::{HashMap, BTreeSet},
	fmt::Debug,
	sync::Arc,
};

/// The material component
#[derive(Debug, Clone)]
pub enum MaterialComponent {
	Texture(Arc<VulkanTexture>),
	Color(Vec4),
	Luminance(f32),
}

/// The legacy illumination model material
#[derive(Default, Debug, Clone)]
pub struct MaterialLegacy {
	/// Base brightness
	pub ambient: MaterialComponent,

	/// Base color
	pub diffuse: MaterialComponent,

	/// Specular color
	pub specular: MaterialComponent,

	/// Specular power
	pub specular_power: MaterialComponent,

	/// Normal map
	pub normal: MaterialComponent,

	/// Emissive, self-lighting
	pub emissive: MaterialComponent,

	/// The other type of components
	pub others: HashMap<String, MaterialComponent>,
}

/// The physically based rendering illumination model material
#[derive(Default, Debug, Clone)]
pub struct MaterialPbr {
	/// Base color
	pub albedo: MaterialComponent,

	/// Normal map
	pub normal: MaterialComponent,

	/// Ambient occlusion
	pub ao: MaterialComponent,

	/// A.k.a. Height map. The renderer must render this map by extruding the mesh, or use ray-marching to cast the protrusions of the map
	pub displacement: MaterialComponent,

	/// Roughness, something sort of the legacy specular-key map
	pub roughness: MaterialComponent,

	/// Metalness, something sort of the legacy specular map
	pub metalness: MaterialComponent,

	/// Emissive, self-lighting
	pub emissive: MaterialComponent,

	/// The other type of components
	pub others: HashMap<String, MaterialComponent>,
}

impl Default for MaterialComponent {
	fn default() -> Self {
		Self::Color(Vec4::new(0.5, 0.5, 0.5, 1.0))
	}
}

/// The `Material` trait helps the `MaterialLegacy` struct or the `MaterialPbr` struct to be able to turn into an object
pub trait Material: Debug {
	/// Get the ambient color
	fn get_ambient(&self) -> Option<&MaterialComponent>;

	/// Get the diffuse color
	fn get_diffuse(&self) -> Option<&MaterialComponent>;

	/// Get the specular color
	fn get_specular(&self) -> Option<&MaterialComponent>;

	/// Get the specular power
	fn get_specular_power(&self) -> Option<&MaterialComponent>;

	/// Get the base color (PBR)
	fn get_albedo(&self) -> Option<&MaterialComponent>;

	/// Get the ambient occlusion (PBR)
	fn get_ao(&self) -> Option<&MaterialComponent>;

	/// Get the displacement map (A.k.a. height map) (PBR)
	fn get_displacement(&self) -> Option<&MaterialComponent>;

	/// Get the roughness map (PBR)
	fn get_roughness(&self) -> Option<&MaterialComponent>;

	/// Get the metalness map (PBR)
	fn get_metalness(&self) -> Option<&MaterialComponent>;

	/// Get the normal map
	fn get_normal(&self) -> Option<&MaterialComponent>;

	/// Get the emissive color
	fn get_emissive(&self) -> Option<&MaterialComponent>;

	/// Get all of the component names exists in this material
	fn get_names(&self) -> BTreeSet<String>;

	/// Get a component by the name of the component
	fn get_by_name(&self, name: &str) -> Option<&MaterialComponent>;

	/// Set a componnet by the name of the component
	fn set_by_name(&mut self, name: &str, texture: MaterialComponent);
}

impl Material for MaterialLegacy {
	fn get_ambient(&self) ->		Option<&MaterialComponent> {Some(&self.ambient)}
	fn get_diffuse(&self) ->		Option<&MaterialComponent> {Some(&self.diffuse)}
	fn get_specular(&self) ->		Option<&MaterialComponent> {Some(&self.specular)}
	fn get_specular_power(&self) ->	Option<&MaterialComponent> {Some(&self.specular_power)}
	fn get_normal(&self) ->			Option<&MaterialComponent> {Some(&self.normal)}
	fn get_emissive(&self) ->		Option<&MaterialComponent> {Some(&self.emissive)}

	fn get_albedo(&self) ->			Option<&MaterialComponent> {None}
	fn get_ao(&self) ->				Option<&MaterialComponent> {None}
	fn get_displacement(&self) ->	Option<&MaterialComponent> {None}
	fn get_roughness(&self) ->		Option<&MaterialComponent> {None}
	fn get_metalness(&self) ->		Option<&MaterialComponent> {None}

	fn get_names(&self) -> BTreeSet<String> {
		let mut ret = BTreeSet::new();
		ret.insert("ambient".to_owned());
		ret.insert("diffuse".to_owned());
		ret.insert("specular".to_owned());
		ret.insert("specular_power".to_owned());
		ret.insert("normal".to_owned());
		ret.insert("emissive".to_owned());
		for (name, _) in self.others.iter() {
			ret.insert(name.clone());
		}
		ret
	}

	fn get_by_name(&self, name: &str) -> Option<&MaterialComponent> {
		match self.others.get(name) {
			Some(data) => Some(data),
			None => {
				match name {
					"ambient" =>		self.get_ambient(),
					"diffuse" =>		self.get_diffuse(),
					"specular" =>		self.get_specular(),
					"specular_power" =>	self.get_specular_power(),
					"normal" =>			self.get_normal(),
					"emissive" =>		self.get_emissive(),
					_ => None,
				}
			}
		}
	}

	fn set_by_name(&mut self, name: &str, texture: MaterialComponent) {
		match name {
			"ambient" =>		self.ambient = texture,
			"diffuse" =>		self.diffuse = texture,
			"specular" =>		self.specular = texture,
			"specular_power" =>	self.specular_power = texture,
			"normal" =>			self.normal = texture,
			"emissive" =>		self.emissive = texture,
			others =>{
				self.others.insert(others.to_owned(), texture);
			}
		}
	}
}

impl Material for MaterialPbr {
	fn get_albedo(&self) ->			Option<&MaterialComponent> {Some(&self.albedo)}
	fn get_ao(&self) ->				Option<&MaterialComponent> {Some(&self.ao)}
	fn get_displacement(&self) ->	Option<&MaterialComponent> {Some(&self.displacement)}
	fn get_roughness(&self) ->		Option<&MaterialComponent> {Some(&self.roughness)}
	fn get_metalness(&self) ->		Option<&MaterialComponent> {Some(&self.metalness)}
	fn get_normal(&self) ->			Option<&MaterialComponent> {Some(&self.normal)}
	fn get_emissive(&self) ->		Option<&MaterialComponent> {Some(&self.emissive)}

	fn get_ambient(&self) ->		Option<&MaterialComponent> {None}
	fn get_diffuse(&self) ->		Option<&MaterialComponent> {None}
	fn get_specular(&self) ->		Option<&MaterialComponent> {None}
	fn get_specular_power(&self) ->	Option<&MaterialComponent> {None}

	fn get_names(&self) -> BTreeSet<String> {
		let mut ret = BTreeSet::new();
		ret.insert("albedo".to_owned());
		ret.insert("ao".to_owned());
		ret.insert("displacement".to_owned());
		ret.insert("roughness".to_owned());
		ret.insert("metalness".to_owned());
		ret.insert("normal".to_owned());
		ret.insert("emissive".to_owned());
		for (name, _) in self.others.iter() {
			ret.insert(name.clone());
		}
		ret
	}

	fn get_by_name(&self, name: &str) -> Option<&MaterialComponent> {
		match self.others.get(name) {
			Some(data) => Some(data),
			None => {
				match name {
					"albedo" =>			self.get_albedo(),
					"ao" =>				self.get_ao(),
					"displacement" =>	self.get_displacement(),
					"roughness" =>		self.get_roughness(),
					"metalness" =>		self.get_metalness(),
					"normal" =>			self.get_normal(),
					"emissive" =>		self.get_emissive(),
					_ => None,
				}
			}
		}
	}

	fn set_by_name(&mut self, name: &str, texture: MaterialComponent) {
		match name {
			"albedo" =>			self.albedo = texture,
			"ao" =>				self.ao = texture,
			"displacement" =>	self.displacement = texture,
			"roughness" =>		self.roughness = texture,
			"metalness" =>		self.metalness = texture,
			"normal" =>			self.normal = texture,
			"emissive" =>		self.emissive = texture,
			others =>{
				self.others.insert(others.to_owned(), texture);
			}
		}
	}
}
