/// Small stack-allocated vector that have interface
/// similar to Rust's Vec<T>.
pub struct SmallVec {
    storage: [u8; Self::MAX_SIZE],
    len: u8,
}

impl SmallVec {
    const MAX_SIZE: usize = 16;

    pub fn new() -> Self {
        SmallVec { storage: [0; 16], len: 0 }
    }

    /// return & reference to written part of underlying buffer
    pub fn as_slice(&self) -> &[u8] {
        return &self.storage[0..self.len as usize];
    }

    /// returns &mut reference to whole underlying buffer
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        return &mut self.storage;
    }

    pub fn seek(&mut self, i: usize) {
        self.len += i as u8;
    }

    pub fn push(&mut self, value: u8) {
        self.storage[self.len as usize] += value;
        self.len += 1;
    }

    pub fn read(&mut self) -> u8 {
        let v = self.storage[self.len as usize];
        self.len += 1;
        return v;
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }
}