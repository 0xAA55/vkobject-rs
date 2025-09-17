
use crate::prelude::*;
use struct_iterable::Iterable;

#[derive(Debug, Clone)]
	pub primitive_type: VkPrimitiveTopology,
}

#[derive(Default, Debug, Clone, Copy, Iterable)]
pub struct UnusedBufferItem {}

pub type UnusedBufferType = BufferVec<UnusedBufferItem>;

pub fn buffer_unused() -> Option<UnusedBufferType> {
	None
}

where
	V: BufferVecItem,
	E: BufferVecItem,
	I: BufferVecItem,
	C: BufferVecItem {
		Self {
			primitive_type,
			vertices,
			indices,
			instances,
			commands,
		}
	}

	}
}
