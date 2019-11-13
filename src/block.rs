const BLOCK_SIZE: usize = 4096;

enum Error {
    BlockFull
}

pub struct Block {
    used_bytes: u16,
    data: [u8; BLOCK_SIZE - 2],
}

impl Block {
    pub fn is_full(&self) -> bool {
        return (self.data.len() - self.used_bytes as usize) < std::mem::size_of::<f32>();
    }

    pub fn push(&mut self, value: f32) -> Result<(), Error> {
        if self.is_full() {
            return Err(Error::BlockFull);
        }

        self.push_bytes(&f32::to_le_bytes(value));
        Ok(())
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        for x in bytes {
            self.data[self.used_bytes as usize] = *x;
            self.used_bytes += 1;
        }
    }

    pub fn iter(&self) -> Iter {
        Iter { block: &self, pos: 0 }
    }
}

struct Iter<'a> {
    block: &'a Block,
    pos: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.block.used_bytes as usize {
            return None;
        }

        let data = [
            self.block.data[self.pos],
            self.block.data[self.pos + 1],
            self.block.data[self.pos + 2],
            self.block.data[self.pos + 3],
        ];

        self.pos += 4;
        Some(f32::from_le_bytes(data))
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq)]
struct BlockHeader {
    loaded_block_id: usize,
}
