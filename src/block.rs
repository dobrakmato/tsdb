use static_assertions::{assert_eq_align, assert_eq_size};

const BLOCK_SIZE: usize = 4096;

/// Header of each block.
struct BlockHeader {
    free_bytes: u8,
}

const BLOCK_HEADER_SIZE: usize = std::mem::size_of::<BlockHeader>();

/// Simple new-type struct representing the block.
struct Block(BlockHeader, [u8; BLOCK_SIZE - BLOCK_HEADER_SIZE]);

// These assertions ensure that we can safely interpret
// any [u8; 4096] as Block and vice-versa.
assert_eq_size!(Block, [u8; BLOCK_SIZE]);
assert_eq_align!(Block, u8);

impl Block {
    fn as_slice(&self) -> &[u8] {
        // Safe: because Block has size of BLOCK_SIZE and alignment of 1.
        let ptr = self as *const Block as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, BLOCK_SIZE) }
    }

    fn as_slice_mut(&mut self) -> &mut [u8] {
        // Safe: because Block has size of BLOCK_SIZE and alignment of 1.
        let ptr = self as *mut Block as *mut u8;
        unsafe { std::slice::from_raw_parts_mut(ptr, BLOCK_SIZE) }
    }
}