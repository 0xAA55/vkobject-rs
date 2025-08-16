
#![allow(unused_imports)]

/// The VkCore initializer
pub mod init;

/// The common things for you to use
pub mod prelude {
	pub use vkcore_rs::*;
	pub use crate::init::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
    }
}
