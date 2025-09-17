
use crate::prelude::*;
use bitvec::vec::BitVec;
use std::{
	cmp::min,
	fmt::{self, Debug, Formatter},
	marker::PhantomData,
	mem::size_of,
	ops::{Index, IndexMut, Range, RangeFrom, RangeTo, RangeFull, RangeInclusive, RangeToInclusive},
	ptr::{copy, null_mut},
	slice::{from_raw_parts, from_raw_parts_mut},
	sync::Arc,
};

/// The type that could be the item of the `BufferVec`
pub trait BufferVecItem: Copy + Sized + Default + Debug {}
impl<T> BufferVecItem for T where T: Copy + Sized + Default + Debug {}

/// The advanced buffer object that could be used as a vector
pub struct BufferVec<T: BufferVecItem> {
	/// The buffer
	buffer: Buffer,

	/// The address of the data in the staging buffer
	staging_buffer_data_address: *mut T,

	/// Num items in the buffer
	num_items: usize,

	/// The capacity of the buffer
	capacity: usize,

	/// The bitmap indicating that the cached (the staging buffer) item was changed
	cache_modified_bitmap: BitVec,

	/// The bitmap indicating that the cached (the staging buffer) data was changed
	cache_modified: bool,

	/// The phantom data to hold the generic type `T`
	_phantom: PhantomData<T>,
}

impl<T> BufferVec<T>
where
	T: BufferVecItem {
	/// Create the `BufferVec<T>`
	pub fn new(device: Arc<VulkanDevice>, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let buffer = Buffer::new(device, 0, None, usage)?;
		Ok(Self {
			buffer,
			staging_buffer_data_address: null_mut(),
			num_items: 0,
			capacity: 0,
			cache_modified_bitmap: BitVec::new(),
			cache_modified: false,
			_phantom: PhantomData,
		})
	}

	/// Get the VkBuffer
	pub(crate) fn get_vk_buffer(&self) -> VkBuffer {
		self.buffer.get_vk_buffer()
	}

	/// Create the `BufferVec<T>` with an initial capacity
	pub fn with_capacity(device: Arc<VulkanDevice>, capacity: usize, usage: VkBufferUsageFlags) -> Result<Self, VulkanError> {
		let mut buffer = Buffer::new(device, capacity as VkDeviceSize, None, usage)?;
		buffer.ensure_staging_buffer()?;
		let staging_buffer_data_address = buffer.get_staging_buffer_address()? as *mut T;
		Ok(Self {
			buffer,
			staging_buffer_data_address,
			num_items: 0,
			capacity,
			cache_modified_bitmap: BitVec::with_capacity(capacity),
			cache_modified: true,
			_phantom: PhantomData,
		})
	}

	/// Change the capacity
	/// * If the capacity is less than the current items, the number of items will be reduced to the new capacity.
	pub fn change_capacity(&mut self, new_capacity: usize) -> Result<(), VulkanError> {
		let mut new_buffer = Buffer::new(self.buffer.device.clone(), new_capacity as VkDeviceSize, None, self.buffer.get_usage())?;
		if new_capacity != 0 {
			let new_address = new_buffer.get_staging_buffer_address()? as *mut T;
			unsafe {copy(self.staging_buffer_data_address as *const T, new_address, self.capacity)}
			self.staging_buffer_data_address = new_address;
			self.cache_modified = true;
			self.cache_modified_bitmap.resize(new_capacity, false);
		} else {
			self.staging_buffer_data_address = null_mut();
			self.cache_modified = false;
			self.cache_modified_bitmap.clear();
			self.cache_modified_bitmap.shrink_to_fit();
		}
		self.buffer = new_buffer;
		self.capacity = new_capacity;
		self.num_items = min(self.num_items, new_capacity);
		Ok(())
	}

	/// Change the length
	/// Forces the length of the vector to new_len.
	///
	/// This is a low-level operation that maintains none of the normal invariants of the type.
	///
	/// # Safety
	///
	/// `new_len` must be less than or equal to `capacity()`.
	/// The elements at `old_len..new_len` must be initialized.
	pub unsafe fn set_len(&mut self, new_len: usize) {
		if new_len > self.num_items {
			for i in self.num_items..new_len {
				self.cache_modified_bitmap.set(i, true);
			}
		}
		self.num_items = new_len;
	}

	/// Get the inner buffer
	pub fn into_inner(self) -> Buffer {
		self.buffer
	}

	/// Creates a `BufferVec<T>` directly from a buffer, a length, and a capacity.
	///
	/// # Safety
	///
	/// This is highly unsafe, just like the Rust official `Vec<T>::from_raw_parts()`
	/// * Unlike the Rust official `Vec<T>::from_raw_parts()`, capacity is not needed to be provided since it was calculated by `buffer.get_size() / size_of::<T>()`
	/// * `length` must be less than the calculated capacity.
	pub unsafe fn from_raw_parts(mut buffer: Buffer, length: usize) -> Result<Self, VulkanError> {
		let capacity = buffer.get_size() as usize / size_of::<T>();
		buffer.ensure_staging_buffer()?;
		let staging_buffer_data_address = buffer.get_staging_buffer_address()? as *mut T;
		Ok(Self {
			buffer,
			staging_buffer_data_address,
			num_items: length,
			capacity,
			cache_modified_bitmap: BitVec::with_capacity(capacity),
			cache_modified: true,
			_phantom: PhantomData,
		})
	}

	/// Enlarge the capacity of the `BufferVec<T>`
	fn grow(&mut self) -> Result<(), VulkanError> {
		let mut new_capacity = ((self.capacity * 3) >> 1) + 1;
		if new_capacity < self.num_items {
			new_capacity = self.num_items;
		}
		self.change_capacity(new_capacity)
	}

	/// Push data to the buffer
	pub fn push(&mut self, data: T) -> Result<(), VulkanError> {
		if self.num_items >= self.capacity {
			self.grow()?;
		}
		unsafe {*self.staging_buffer_data_address.wrapping_add(self.num_items) = data};
		self.cache_modified = true;
		self.cache_modified_bitmap.push(true);
		self.num_items += 1;
		Ok(())
	}

	/// Pop data from the buffer
	pub fn pop(&mut self) -> T {
		if self.num_items == 0 {
			panic!("`BufferVec::<T>::pop()` called on an empty `BufferVec<T>`.");
		}
		self.num_items -= 1;
		self.cache_modified_bitmap.pop();
		unsafe {*self.staging_buffer_data_address.wrapping_add(self.num_items)}
	}

	/// Removes and returns the element at position index within the vector, shifting all elements after it to the left.
	///
	/// Note: Because this shifts over the remaining elements, it has a worst-case performance of O(n). If you don’t need the order of elements to be preserved, use `swap_remove` instead.
	///
	/// # Panics
	/// Panics if `index` is out of bounds.
	pub fn remove(&mut self, index: usize) -> T {
		let ret = self[index];
		let from_index = index + 1;
		unsafe {copy(
			self.staging_buffer_data_address.wrapping_add(from_index),
			self.staging_buffer_data_address.wrapping_add(index),
			self.num_items - from_index)
		};
		self.num_items -= 1;
		for i in index..self.num_items {
			self.cache_modified_bitmap.set(i, true);
		}
		self.cache_modified_bitmap.pop();
		ret
	}

	/// Removes an element from the vector and returns it.
	///
	/// The removed element is replaced by the last element of the vector.
	///
	/// This does not preserve ordering of the remaining elements, but is O(1). If you need to preserve the element order, use `remove` instead.
	///
	/// # Panics
	/// Panics if `index` is out of bounds.
	pub fn swap_remove(&mut self, index: usize) -> T {
		if self.num_items > 1 {
			let last_index = self.num_items - 1;
			let last_item = unsafe {&mut *self.staging_buffer_data_address.wrapping_add(self.num_items)};
			let swap_item = &mut self[index];
			let ret = *swap_item;
			if last_index != index {
				*swap_item = *last_item;
			}
			self.num_items -= 1;
			self.cache_modified_bitmap.pop();
			ret
		} else {
			if index != 0 {
				panic!("Index {index} out of bounds (len() == {})", self.len());
			}
			self.pop()
		}
	}

	/// Resize the buffer
	pub fn resize(&mut self, new_len: usize, new_data: T) -> Result<(), VulkanError> {
		if self.num_items == new_len && self.capacity >= self.num_items {
			return Ok(());
		}
		self.cache_modified = true;
		if self.capacity < new_len {
			self.change_capacity(new_len)?;
		}
		if new_len > self.num_items {
			self.cache_modified = true;
			unsafe {from_raw_parts_mut(self.staging_buffer_data_address.wrapping_add(self.num_items), new_len - self.num_items)}.fill(new_data);
			for i in self.num_items..new_len {
				self.cache_modified_bitmap.set(i, true);
			}
		}
		self.num_items = new_len;
		self.cache_modified_bitmap.resize(new_len, false);
		Ok(())
	}

	/// Clear the buffer
	pub fn clear(&mut self) {
		self.num_items = 0;
	}

	/// Get the capacity
	pub fn get_capacity(&self) -> usize {
		self.capacity
	}

	/// Get num items in the buffer
	pub fn len(&self) -> usize {
		self.num_items
	}

	/// Get is the buffer empty?
	pub fn is_empty(&self) -> bool {
		self.num_items == 0
	}

	/// Shrink to fit
	pub fn shrink_to_fit(&mut self) -> Result<(), VulkanError> {
		self.change_capacity(self.num_items)
	}

	/// Flush the staging buffer to the device memory
	pub fn flush(&mut self, cmdbuf: VkCommandBuffer) -> Result<(), VulkanError> {
		if !self.cache_modified {
			return Ok(());
		}
		const MAX_GAP: usize = 16;
		let mut si = 0;
		let mut ei = 0;
		let mut gap = 0;
		let mut is_in = false;
		let mut region: Vec<BufferRegion> = Vec::new();
		for (i, b) in self.cache_modified_bitmap.iter().enumerate() {
			if *b {
				if !is_in {
					is_in = true;
					si = i;
					gap = 0;
				}
			} else if is_in {
				ei = i;
				is_in = false;
				gap = 1; // This ensures all regions were flushed including the last one.
			} else {
				gap += 1;
				if gap == MAX_GAP {
					region.push(BufferRegion {
						offset: (si * size_of::<T>()) as VkDeviceSize,
						size: ((ei + 1 - si) * size_of::<T>()) as VkDeviceSize,
					});
				}
			}
		}
		self.cache_modified_bitmap.fill(false);
		if is_in || gap != 0 {
			region.push(BufferRegion {
				offset: (si * size_of::<T>()) as VkDeviceSize,
				size: ((ei + 1 - si) * size_of::<T>()) as VkDeviceSize,
			});
		}
		if !region.is_empty() {
			self.buffer.upload_staging_buffer_multi(cmdbuf, region.as_ref())?;
		}
		self.cache_modified = false;
		Ok(())
	}
}

impl<T> Clone for BufferVec<T>
where
	T: BufferVecItem {
	fn clone(&self) -> Self {
		let mut buffer = self.buffer.clone();
		let staging_buffer_data_address = buffer.get_staging_buffer_address().unwrap() as *mut T;
		Self {
			buffer,
			staging_buffer_data_address,
			num_items: self.num_items,
			capacity: self.capacity,
			cache_modified_bitmap: self.cache_modified_bitmap.clone(),
			cache_modified: self.cache_modified,
			_phantom: self._phantom,
		}
	}
}

impl<T> Debug for BufferVec<T>
where
	T: BufferVecItem {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		f.debug_struct("BufferVec")
		.field("buffer", &self.buffer)
		.field("staging_buffer_data_address", &self.staging_buffer_data_address)
		.field("num_items", &self.num_items)
		.field("capacity", &self.capacity)
		.field("cache_modified_bitmap", &self.cache_modified_bitmap)
		.field("cache_modified", &self.cache_modified)
		.finish()
	}
}

impl<T> Index<usize> for BufferVec<T>
where
	T: BufferVecItem {
	type Output = T;
	fn index(&self, index: usize) -> &T {
		if index >= self.len() {
			panic!("Index {index:?} out of bounds (len() == {})", self.len());
		}
		unsafe {&*self.staging_buffer_data_address.wrapping_add(index)}
	}
}

impl<T> IndexMut<usize> for BufferVec<T>
where
	T: BufferVecItem {
	fn index_mut(&mut self, index: usize) -> &mut T {
		if index >= self.len() {
			panic!("Index {index:?} out of bounds (len() == {})", self.len());
		}
		self.cache_modified = true;
		self.cache_modified_bitmap.set(index, true);
		unsafe {&mut *self.staging_buffer_data_address.wrapping_add(index)}
	}
}

impl<T> Index<Range<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	type Output = [T];
	fn index(&self, range: Range<usize>) -> &[T] {
		if range.start >= self.len() && range.end > self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		unsafe {from_raw_parts(self.staging_buffer_data_address.wrapping_add(range.start), range.end - range.start)}
	}
}

impl<T> IndexMut<Range<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	fn index_mut(&mut self, range: Range<usize>) -> &mut [T] {
		if range.start >= self.len() && range.end > self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		self.cache_modified = true;
		for i in range.clone() {
			self.cache_modified_bitmap.set(i, true);
		}
		unsafe {from_raw_parts_mut(self.staging_buffer_data_address.wrapping_add(range.start), range.end - range.start)}
	}
}

impl<T> Index<RangeFrom<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	type Output = [T];
	fn index(&self, range: RangeFrom<usize>) -> &[T] {
		if range.start >= self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		unsafe {from_raw_parts(self.staging_buffer_data_address.wrapping_add(range.start), self.len() - range.start)}
	}
}

impl<T> IndexMut<RangeFrom<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut [T] {
		if range.start >= self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		self.cache_modified = true;
		for i in range.start..self.len() {
			self.cache_modified_bitmap.set(i, true);
		}
		unsafe {from_raw_parts_mut(self.staging_buffer_data_address.wrapping_add(range.start), self.len() - range.start)}
	}
}

impl<T> Index<RangeTo<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	type Output = [T];
	fn index(&self, range: RangeTo<usize>) -> &[T] {
		if range.end > self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		unsafe {from_raw_parts(self.staging_buffer_data_address, range.end)}
	}
}

impl<T> IndexMut<RangeTo<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	fn index_mut(&mut self, range: RangeTo<usize>) -> &mut [T] {
		if range.end > self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		self.cache_modified = true;
		for i in 0..range.end {
			self.cache_modified_bitmap.set(i, true);
		}
		unsafe {from_raw_parts_mut(self.staging_buffer_data_address, range.end)}
	}
}

impl<T> Index<RangeFull> for BufferVec<T>
where
	T: BufferVecItem {
	type Output = [T];
	fn index(&self, _: RangeFull) -> &[T] {
		unsafe {from_raw_parts(self.staging_buffer_data_address, self.len())}
	}
}

impl<T> IndexMut<RangeFull> for BufferVec<T>
where
	T: BufferVecItem {
	fn index_mut(&mut self, _: RangeFull) -> &mut [T] {
		self.cache_modified = true;
		self.cache_modified_bitmap.fill(true);
		unsafe {from_raw_parts_mut(self.staging_buffer_data_address, self.len())}
	}
}

impl<T> Index<RangeInclusive<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	type Output = [T];
	fn index(&self, range: RangeInclusive<usize>) -> &[T] {
		if *range.start() >= self.len() || *range.end() >= self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		unsafe {from_raw_parts(self.staging_buffer_data_address.wrapping_add(*range.start()), range.end() + 1 - range.start())}
	}
}

impl<T> IndexMut<RangeInclusive<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	fn index_mut(&mut self, range: RangeInclusive<usize>) -> &mut [T] {
		if *range.start() >= self.len() || *range.end() >= self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		self.cache_modified = true;
		for i in range.clone() {
			self.cache_modified_bitmap.set(i, true);
		}
		unsafe {from_raw_parts_mut(self.staging_buffer_data_address.wrapping_add(*range.start()), range.end() + 1 - range.start())}
	}
}

impl<T> Index<RangeToInclusive<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	type Output = [T];
	fn index(&self, range: RangeToInclusive<usize>) -> &[T] {
		if range.end >= self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		unsafe {from_raw_parts(self.staging_buffer_data_address, range.end + 1)}
	}
}

impl<T> IndexMut<RangeToInclusive<usize>> for BufferVec<T>
where
	T: BufferVecItem {
	fn index_mut(&mut self, range: RangeToInclusive<usize>) -> &mut [T] {
		if range.end >= self.len() {
			panic!("Slice range {range:?} out of bounds (len() == {})", self.len());
		}
		self.cache_modified = true;
		for i in 0..=range.end {
			self.cache_modified_bitmap.set(i, true);
		}
		unsafe {from_raw_parts_mut(self.staging_buffer_data_address, range.end + 1)}
	}
}
