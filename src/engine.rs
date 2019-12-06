use crate::engine::array_vec::ArrayVec;
use std::ops::Sub;
use serde::{Serialize, Deserialize};

/// Represents UNIX timestamp as number of seconds from the
/// start of the epoch.
///
/// Timestamps can be subtracted and converted into u64.
#[derive(Eq, PartialEq, Debug, Copy, Clone, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Timestamp(u64);

impl Into<Timestamp> for u64 {
    fn into(self) -> Timestamp {
        Timestamp(self)
    }
}

impl Into<u64> for &Timestamp {
    fn into(self) -> u64 {
        self.0
    }
}

impl Sub<Timestamp> for Timestamp {
    type Output = u64;

    fn sub(self, rhs: Timestamp) -> Self::Output {
        self.0 - rhs.0
    }
}

/// Represents a single data-point with associated Timestamp
/// in Series object.
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct Point<V> {
    timestamp: Timestamp,
    value: V,
}

impl<V> Point<V> {
    #[inline]
    pub fn timestamp(&self) -> &Timestamp {
        &self.timestamp
    }

    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
}


pub trait Decoder {
    fn new(min_timestamp: Timestamp) -> Self;
}

pub trait Schema<T> {
    type EncState: Default;
    type DecState: Decoder;
    type Schema: Default;
    fn encode(state: &mut Self::EncState, point: Point<T>) -> ArrayVec<u8>;
    fn decode(state: &mut Self::DecState, buff: &[u8]) -> (Point<T>, usize);
}

pub mod array_vec;

pub mod f32 {
    use crate::engine::{Schema, Point, Decoder, Timestamp};
    use nano_leb128::ULEB128;
    use crate::engine::array_vec::ArrayVec;
    use serde::{Serialize, Deserialize};

    pub struct F32;

    #[derive(Serialize, Deserialize, Copy, Clone)]
    pub struct F32Enc(Timestamp);

    impl Default for F32Enc {
        fn default() -> Self {
            F32Enc(0.into())
        }
    }

    #[derive(Serialize, Deserialize, Copy, Clone)]
    pub struct F32Dec(u64);

    impl Schema<f32> for F32 {
        type EncState = F32Enc;
        type DecState = F32Dec;
        type Schema = ();

        fn encode(state: &mut Self::EncState, point: Point<f32>) -> ArrayVec<u8> {
            let mut array = ArrayVec::default();
            let dt = point.timestamp - state.0;
            state.0 = point.timestamp;

            array.length += ULEB128::from(dt)
                .write_into(&mut array.data)
                .unwrap();
            array.extend(&point.value.to_le_bytes());

            array
        }

        fn decode(state: &mut Self::DecState, buff: &[u8]) -> (Point<f32>, usize) {
            let (dt, read) = ULEB128::read_from(buff).unwrap();
            let value = f32::from_le_bytes([
                buff[read + 0],
                buff[read + 1],
                buff[read + 2],
                buff[read + 3]
            ]);

            state.0 = state.0 + u64::from(dt);

            (Point { timestamp: state.0.into(), value }, read + 4)
        }
    }

    impl Decoder for F32Dec {
        fn new(min_timestamp: Timestamp) -> Self {
            F32Dec(min_timestamp.0)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::engine::{Point, Schema, Timestamp, Decoder};
        use crate::engine::f32::{F32Enc, F32Dec, F32};

        #[test]
        fn test_encode_decode() {
            let point1 = Point { timestamp: 40.into(), value: 3.14 };
            let point2 = Point { timestamp: 120.into(), value: 10.0 };

            let mut enc_state = F32Enc::default();
            let mut dec_state = F32Dec::new(Timestamp(0));

            let result1 = F32::encode(&mut enc_state, point1);
            let result2 = F32::encode(&mut enc_state, point2);

            let (decoded1, _) = F32::decode(&mut dec_state, &result1.data);
            let (decoded2, _) = F32::decode(&mut dec_state, &result2.data);

            assert_eq!(decoded1, point1);
            assert_eq!(decoded2, point2);
        }
    }
}

pub mod block {
    use static_assertions::{assert_eq_align, assert_eq_size};

    /// Type representing metadata about the Block.
    pub struct BlockTail {
        free_bytes: u8,
    }

    pub const BLOCK_SIZE: usize = 4096;
    pub const BLOCK_TAIL_SIZE: usize = std::mem::size_of::<BlockTail>();

    /// Type owning data of a block.
    pub struct Block {
        pub data: [u8; BLOCK_SIZE - BLOCK_TAIL_SIZE],
        tail: BlockTail,
    }

    // These assertions ensure that we can safely interpret
    // any [u8; 4096] as Block and vice-versa.
    assert_eq_size!(Block, [u8; BLOCK_SIZE]);
    assert_eq_align!(Block, u8);

    impl Block {
        /// Stores specified value as free bytes inside the header of this
        /// block.
        pub fn set_free_bytes(&mut self, val: u8) {
            self.tail.free_bytes = val;
        }

        /// Returns the length of data in bytes. This function only returns
        /// valid result when the Block is closed.
        #[inline]
        pub fn data_len(&self) -> usize {
            (BLOCK_SIZE - BLOCK_TAIL_SIZE) - self.tail.free_bytes as usize
        }

        /// Returns the byte representation of this block including the header
        /// as immutable slice of bytes.
        pub fn as_slice(&self) -> &[u8] {
            // Safe: because Block has size of BLOCK_SIZE and alignment of 1.
            let ptr = self as *const Block as *const u8;
            unsafe { std::slice::from_raw_parts(ptr, BLOCK_SIZE) }
        }

        /// Returns the byte representation of this block including the header
        /// as mutable slice of bytes.
        pub fn as_slice_mut(&mut self) -> &mut [u8] {
            // Safe: because Block has size of BLOCK_SIZE and alignment of 1.
            let ptr = self as *mut Block as *mut u8;
            unsafe { std::slice::from_raw_parts_mut(ptr, BLOCK_SIZE) }
        }
    }

    impl Default for Block {
        fn default() -> Self {
            Block {
                tail: BlockTail { free_bytes: 0 },
                data: [0; BLOCK_SIZE - BLOCK_TAIL_SIZE],
            }
        }
    }

    /// Type that represents a "handle" to potentially unloaded Block.
    #[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
    pub struct BlockSpec {
        pub series_id: usize,
        pub block_id: usize,
    }
}

pub mod cache {
    use log::trace;
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::fmt::Debug;

    /// An LRU cache backed by HashMap which owns the stored data. The
    /// least recently used algorithm is implemented using runtime
    /// allocated ring buffer as circular queue.
    pub struct Cache<K, V> {
        items: HashMap<K, V>,
        lru: Vec<Option<K>>,
        lru_idx: usize,
    }

    impl<K, V> Cache<K, V> where K: Hash + Eq + Copy + Debug {
        pub fn with_capacity(capacity: usize) -> Self {
            Cache {
                items: HashMap::with_capacity(capacity),
                lru: vec![None; capacity],
                lru_idx: 0,
            }
        }

        pub fn contains_key(&self, key: &K) -> bool {
            self.items.contains_key(key)
        }

        pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
            self.items.get_mut(key)
        }

        fn evict_lru(&mut self) {
            if let Some(t) = self.lru[self.lru_idx] {
                self.remove(&t);
                self.lru[self.lru_idx] = None;
            }
        }

        pub fn insert(&mut self, key: K, value: V) {
            self.evict_lru();

            self.items.insert(key, value);
            self.lru[self.lru_idx] = Some(key);
            self.lru_idx = (self.lru_idx + 1) % self.lru.len();
        }

        /// Elements inserted with this method are not counted towards the
        /// cache capacity so it's possible to have more then `capacity`
        /// elements in the cache.
        pub fn insert_unevictable(&mut self, key: K, value: V) where K: Debug {
            trace!("Inserted unevictable entry: {:?}", &key);
            self.items.insert(key, value);
        }

        pub fn remove(&mut self, key: &K) where K: Debug {
            trace!("Removed entry: {:?}", key);
            self.items.remove(key);
        }

        pub fn len(&self) -> usize {
            self.items.len()
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::engine::cache::Cache;

        #[test]
        fn test_get_mut() {
            let mut c = Cache::<u8, u8>::with_capacity(8);
            c.insert(0, 65);
            c.insert(1, 60);

            let mut a = c.get_mut(&0);
            assert_eq!(*a.unwrap(), 65);
        }

        #[test]
        fn test_remove() {
            let mut c = Cache::<u8, u8>::with_capacity(8);
            c.insert(0, 65);
            c.remove(&0);

            assert_eq!(c.len(), 0);
        }

        #[test]
        fn test_contains_without_eviction() {
            let mut c = Cache::<u8, u8>::with_capacity(8);
            c.insert(0, 60);
            c.insert(1, 60);
            c.insert(2, 60);

            assert_eq!(c.len(), 3);
            assert!(c.contains_key(&0));
            assert!(c.contains_key(&1));
            assert!(c.contains_key(&2));
            assert!(!c.contains_key(&3));
        }

        #[test]
        fn test_eviction() {
            let mut c = Cache::<u8, u8>::with_capacity(3);

            c.insert(0, 60);
            c.insert(1, 60);
            c.insert(2, 60);

            assert_eq!(c.len(), 3);
            assert!(c.contains_key(&0));
            assert!(c.contains_key(&1));
            assert!(c.contains_key(&2));
            assert!(!c.contains_key(&3));

            c.insert(3, 70);
            assert!(c.contains_key(&3));
            assert!(!c.contains_key(&0));

            c.insert_unevictable(5, 50);
            assert!(c.contains_key(&1));
            assert!(c.contains_key(&2));
            assert!(c.contains_key(&3));
            assert!(c.contains_key(&5));
        }
    }
}

pub mod io {
    use log::debug;
    use std::path::{Path, PathBuf};
    use crate::engine::block::{Block, BlockSpec, BLOCK_SIZE};
    use std::collections::HashMap;
    use std::fs::{File, OpenOptions};
    use std::collections::hash_map::Entry;
    use std::io::{Write, Seek, SeekFrom, Read, Error, ErrorKind};
    use serde::{Serialize, Deserialize};

    #[derive(Copy, Clone, Debug)]
    #[derive(Serialize, Deserialize)]
    pub enum SyncPolicy {
        Immediate,
        Never,
    }

    struct StorageFile {
        file: File,
        length: usize,
    }

    impl StorageFile {
        const INITIAL_ALLOCATION_SIZE: usize = 2 * BLOCK_SIZE;

        /// Creates a StorageFile by creation a file on the disk and allocates
        /// the storage by filling the file with zeros.
        fn allocate_new(path: &Path) -> Result<StorageFile, Error> {
            let mut file = OpenOptions::new()
                .write(true)
                .read(true)
                .create_new(true)
                .open(path)?;

            // allocate space for some blocks
            file.write_all(&[0u8; StorageFile::INITIAL_ALLOCATION_SIZE])?;

            Ok(StorageFile { file, length: StorageFile::INITIAL_ALLOCATION_SIZE })
        }

        /// Opens the file on the disk and returns a new StorageFile instance
        /// containing the opened file.
        fn open(path: &Path) -> Result<StorageFile, Error> {
            let file = OpenOptions::new()
                .write(true)
                .read(true)
                .open(path)?;

            let metadata = file.metadata()?;

            if (metadata.len() % BLOCK_SIZE as u64) != 0 {
                return Err(Error::new(ErrorKind::Other, "File length is not multiple of BLOCK_SIZE!"));
            }

            Ok(StorageFile { file, length: metadata.len() as usize })
        }

        fn ensure_file_is_large_enough(&mut self, spec: &BlockSpec) -> Result<(), Error> {
            let start = spec.starting_pos_in_file();

            // if the highest index in the file is smaller than
            // the starting position of this block, we must allocate the data
            if self.length - 1 < start {
                self.file.seek(SeekFrom::End(0))?;

                while self.length - 1 < start {
                    self.file.write_all(&[0u8; BLOCK_SIZE])?;
                    self.length += BLOCK_SIZE;
                }
            }

            Ok(())
        }

        /// Loads data for block specified by BlockSpec into passed in
        /// mutable Block storage.
        ///
        /// It the file currently does not contain specified Block the space
        /// needed to store the block is allocated.
        fn read_into(&mut self, spec: &BlockSpec, block: &mut Block) -> Result<(), Error> {
            self.ensure_file_is_large_enough(spec)?;

            let seek = SeekFrom::Start(spec.starting_pos_in_file() as u64);
            self.file.seek(seek)?;
            self.file.read_exact(block.as_slice_mut())
        }

        /// Writes part of block or whole block (if `dirty_from == 0`) to the
        /// file by seeking to `block_start + dirty_from` and writing the part
        /// of block that is dirty.
        fn write(&mut self, spec: &BlockSpec, block: &Block, dirty_from: usize) -> Result<(), Error> {
            let seek = SeekFrom::Start(spec.starting_pos_in_file() as u64 + dirty_from as u64);

            self.file.seek(seek)?;
            self.file.write_all(&block.as_slice()[dirty_from..])
        }

        /// Syncs the file's contents (not necessarily metadata) to the disk
        /// to ensure all the data has been written.
        fn sync(&mut self) -> Result<(), Error> {
            self.file.sync_data()
        }
    }

    pub struct BlockIO {
        open_files: HashMap<usize, StorageFile>,
        sync_policy: SyncPolicy,
        storage: PathBuf,
    }

    const BLOCKS_PER_FILE: usize = 8192;

    impl BlockSpec {
        /// Returns file id this block resides in. The code in this function
        /// is equivalent to `self.block_id / BLOCKS_PER_FILE`.
        #[inline]
        pub fn file_id(&self) -> usize {
            self.block_id / BLOCKS_PER_FILE
        }

        /// Returns the offset in bytes inside the file the Block specified
        /// by this BlockSpec resides in.
        pub fn starting_pos_in_file(&self) -> usize {
            let file_local_block_id = self.block_id - (self.file_id() * BLOCKS_PER_FILE);
            file_local_block_id * BLOCK_SIZE
        }

        /// Returns the file path of the file the Block specified by this
        /// BlockSpec resides in.
        pub fn determine_file_path(&self, base_path: &Path) -> PathBuf {
            base_path.join(format!("{}-{}.dat", self.series_id, self.file_id()))
        }
    }

    impl BlockIO {
        pub fn new(storage: PathBuf, sync_policy: SyncPolicy) -> Self {
            BlockIO {
                open_files: Default::default(),
                sync_policy,
                storage,
            }
        }

        /// Creates or loads the file that contains block specified by `spec`
        /// parameter. This function does not load file's contents into memory.
        fn create_or_load_file(&mut self, spec: &BlockSpec) -> &mut StorageFile {
            // if the file is already open we just return in. in the
            // other case we need to either load the file or create it.
            match self.open_files.entry(spec.file_id()) {
                Entry::Occupied(t) => t.into_mut(),
                Entry::Vacant(t) => {
                    let file_path = spec.determine_file_path(&self.storage);
                    let storage_file = if file_path.exists() {
                        debug!("Loading storage file: {}", file_path.to_string_lossy());
                        StorageFile::open(&file_path).expect("open block failed")
                    } else {
                        debug!("Creating storage file: {}", file_path.to_string_lossy());
                        StorageFile::allocate_new(&file_path).expect("create block failed")
                    };

                    t.insert(storage_file)
                }
            }
        }

        /// This function either creates or loads the block specified by the
        /// BlockSpec to the memory and returns the Block instance owning
        /// the data.
        ///
        /// Each time this function is called a new instance of Block is
        /// created even if the BlockSpec is same.
        pub fn create_or_load_block(&mut self, spec: &BlockSpec) -> Block {
            let mut block = Block::default();

            self.create_or_load_file(spec)
                .read_into(&spec, &mut block)
                .expect("load block failed");

            block
        }

        /// Writes the data of whole Block into file specified by BlockSpec.
        ///
        /// It is preferred to call `write_block_partially` if only part of
        /// block's data has been changed.
        pub fn write_block_fully(&mut self, spec: &BlockSpec, block: &Block) {
            self.write_block_partially(spec, block, 0);
        }

        /// Writes the data of part of the Block into the file specified by
        /// BlockSpec. The written part of the Block is determined by `dirty_from`
        /// parameter which tells how many bytes in the Block are not dirty.
        ///
        /// This allows to write only the dirty part of the Block skipping the
        /// overwriting of the previously written data that is not changed.
        pub fn write_block_partially(&mut self, spec: &BlockSpec, block: &Block, dirty_from: usize) {
            let should_sync = match self.sync_policy {
                SyncPolicy::Immediate => true,
                SyncPolicy::Never => false,
            };
            let file = self.create_or_load_file(spec);
            file.write(&spec, block, dirty_from)
                .expect("write block partially failed");

            if should_sync {
                file.sync().expect("sync failed")
            }
        }
    }
}

pub mod index {
    use log::debug;
    use crate::engine::Timestamp;
    use std::fs::{File, OpenOptions};
    use crate::engine::io::SyncPolicy;
    use std::path::Path;
    use std::io::{Error, Read, Seek, SeekFrom, Write};

    struct MinMax {
        min: Timestamp,
        max: Timestamp,
    }

    pub struct TimestampIndex {
        file: File,
        sync_policy: SyncPolicy,
        data: Vec<MinMax>,
        dirty_from_element_idx: usize,
    }

    impl TimestampIndex {
        /// Loads file specified by path as timestamp index and
        /// returns TimestampIndex containing the data from file.
        fn load(path: &Path, sync_policy: SyncPolicy) -> Result<TimestampIndex, Error> {
            let mut file = OpenOptions::new()
                .write(true)
                .read(true)
                .open(path)?;

            let metadata = file.metadata()?;
            let elements = metadata.len() as usize / std::mem::size_of::<MinMax>();
            let mut data = Vec::with_capacity(elements);

            debug!("Loading index from: {} ({} elements)", path.to_string_lossy(), elements);
            // todo: check if unsafe is safe?
            // safe: we take Vec's allocation, create a slice of u8 from it, read
            // the data into the slice.
            let ptr = data.as_mut_ptr() as *mut u8;
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(ptr, elements * std::mem::size_of::<MinMax>())
            };

            file.read_exact(bytes)?;

            Ok(TimestampIndex {
                file,
                sync_policy,
                dirty_from_element_idx: data.len(),
                data,
            })
        }

        /// Creates a new empty index file and returns TimestampIndex
        /// associated with it.
        fn create(path: &Path, sync_policy: SyncPolicy) -> Result<TimestampIndex, Error> {
            debug!("Creating new index at: {}", path.to_string_lossy());
            let file = OpenOptions::new()
                .write(true)
                .read(true)
                .create_new(true)
                .open(path)?;

            Ok(TimestampIndex { file, sync_policy, data: vec![], dirty_from_element_idx: 0 })
        }

        /// Creates or loads the index file from specified path.
        pub fn create_or_load(path: &Path, sync_policy: SyncPolicy) -> Result<TimestampIndex, Error> {
            if !path.exists() {
                Self::create(path, sync_policy)
            } else {
                Self::load(path, sync_policy)
            }
        }

        /// Ensures that the internal Vec storage is large enough to store
        /// data for block specified by `for_block_id`. If the internal Vec
        /// is not large enough the capacity of Vec is increased by 8 additional
        /// elements.
        ///
        /// This function is safe to call only with `for_block_id` less or equal
        /// to previous `for_block_id + 1`.
        ///
        /// # Panics
        /// This functions panic when `for_block_id` is larger then last
        /// `for_block_id + 1`.
        fn ensure_vec_has_enough_elements(&mut self, for_block_id: usize) {
            assert!(self.data.len() >= for_block_id);

            if self.data.len() < for_block_id + 1 {
                // when we push to full Vec the backing allocation size is
                // doubled.
                //
                // to prevent frequent reallocations and to prevent
                // exponential growth of the index (because it grows
                // linearly) we manually reserve 8 elements each time.
                if self.data.capacity() == self.data.len() {
                    self.data.reserve(8);
                }
                self.data.push(MinMax { min: Timestamp(0), max: Timestamp(0) });
            }
        }

        /// Sets the minimum timestamp for specified block.
        ///
        /// This function is safe to call only with `block_id` less or equal
        /// to previous `block_id + 1`.
        ///
        /// # Panics
        /// This functions panic when `block_id` is larger then last
        /// `block_id + 1`.
        pub fn set_min(&mut self, block_id: usize, timestamp: Timestamp) {
            self.ensure_vec_has_enough_elements(block_id);
            if let Some(t) = self.data.get_mut(block_id) {
                t.min = timestamp
            };
            self.dirty_from_element_idx = self.dirty_from_element_idx.min(block_id);
        }

        /// Sets the maximum timestamp for specified block.
        ///
        /// This function is safe to call only with `block_id` less or equal
        /// to previous `block_id + 1`.
        ///
        /// # Panics
        /// This functions panic when `block_id` is larger then last
        /// `block_id + 1`.
        pub fn set_max(&mut self, block_id: usize, timestamp: Timestamp) {
            self.ensure_vec_has_enough_elements(block_id);
            if let Some(t) = self.data.get_mut(block_id) {
                t.max = timestamp
            };
            self.dirty_from_element_idx = self.dirty_from_element_idx.min(block_id);
        }

        pub fn get_min(&self, block_id: usize) -> &Timestamp {
            &self.data.get(block_id).unwrap().min
        }

        pub fn get_max(&self, block_id: usize) -> &Timestamp {
            &self.data.get(block_id).unwrap().max
        }

        pub fn find_block(&self, timestamp: &Timestamp) -> usize {
            match self.data.binary_search_by(|x| x.max.0.cmp(&timestamp.0)) {
                Ok(mut t) => {
                    if t == 0 {
                        return t;
                    }
                    let t_orig_max = self.get_max(t);

                    while t > 0 && self.get_max(t - 1) == t_orig_max {
                        t -= 1;
                    }
                    t
                }
                Err(t) => t
            }
        }

        pub fn write_dirty_part(&mut self) {
            let dirty_from_bytes = self.dirty_from_element_idx * std::mem::size_of::<MinMax>();

            self.file.seek(SeekFrom::Start(dirty_from_bytes as u64))
                .expect("cannot seek in index file");

            // TODO: check if unsafe is safe
            // safe: we take Vec's allocation, create a slice of u8 from it
            // write the bytes of the slice to file
            let elements = self.data.len();
            let ptr = self.data.as_ptr() as *const u8;
            let bytes = unsafe {
                std::slice::from_raw_parts(ptr, elements * std::mem::size_of::<MinMax>())
            };

            self.file.write(&bytes[dirty_from_bytes..])
                .expect("cannot write to index file");

            if let SyncPolicy::Immediate = &self.sync_policy {
                self.file.sync_data().expect("sync failed")
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::engine::index::TimestampIndex;
        use std::fs::File;
        use tempdir::TempDir;
        use crate::engine::io::SyncPolicy;
        use crate::engine::Timestamp;

        #[test]
        fn test_get_set_max() {
            let dir = TempDir::new("tsdb").unwrap();
            let mut idx = TimestampIndex {
                file: File::create(dir.path().join("test_get_set_max")).unwrap(),
                sync_policy: SyncPolicy::Never,
                data: vec![],
                dirty_from_element_idx: 0,
            };

            idx.set_min(0, Timestamp(5));
            idx.set_min(1, Timestamp(10));
            idx.set_min(2, Timestamp(16));
            idx.set_min(3, Timestamp(20));
            idx.set_min(4, Timestamp(21));
            idx.set_min(5, Timestamp(50));

            assert_eq!(*idx.get_min(0), Timestamp(5));
            assert_eq!(*idx.get_min(4), Timestamp(21));
        }

        #[test]
        fn test_binary_search() {
            let dir = TempDir::new("tsdb").unwrap();
            let mut idx = TimestampIndex {
                file: File::create(dir.path().join("test_binary_search")).unwrap(),
                sync_policy: SyncPolicy::Never,
                data: vec![],
                dirty_from_element_idx: 0,
            };

            // 5 | 10 | 16 | 20 | 20 | 21 | 50

            idx.set_max(0, Timestamp(5));
            idx.set_max(1, Timestamp(10));
            idx.set_max(2, Timestamp(16));
            idx.set_max(3, Timestamp(20));
            idx.set_max(4, Timestamp(20));
            idx.set_max(5, Timestamp(21));
            idx.set_max(6, Timestamp(50));

            assert_eq!(idx.find_block(&Timestamp(0)), (0));
            assert_eq!(idx.find_block(&Timestamp(5)), (0));
            assert_eq!(idx.find_block(&Timestamp(6)), (1));
            assert_eq!(idx.find_block(&Timestamp(18)), (3));
            assert_eq!(idx.find_block(&Timestamp(19)), (3));
            assert_eq!(idx.find_block(&Timestamp(20)), (3));
            assert_eq!(idx.find_block(&Timestamp(21)), (5));
            assert_eq!(idx.find_block(&Timestamp(25)), (6));
            assert_eq!(idx.find_block(&Timestamp(1000)), (7));
        }

        #[test]
        fn test_binary_search_same() {
            let dir = TempDir::new("tsdb").unwrap();
            let mut idx = TimestampIndex {
                file: File::create(dir.path().join("test_binary_search_same")).unwrap(),
                sync_policy: SyncPolicy::Never,
                data: vec![],
                dirty_from_element_idx: 0,
            };

            idx.set_max(0, Timestamp(5));
            idx.set_max(1, Timestamp(5));
            idx.set_max(2, Timestamp(5));
            idx.set_max(3, Timestamp(10));
            idx.set_max(4, Timestamp(10));

            assert_eq!(idx.find_block(&Timestamp(0)), (0));
            assert_eq!(idx.find_block(&Timestamp(5)), (0));
            assert_eq!(idx.find_block(&Timestamp(6)), (3));
            assert_eq!(idx.find_block(&Timestamp(7)), (3));
            assert_eq!(idx.find_block(&Timestamp(8)), (3));
            assert_eq!(idx.find_block(&Timestamp(9)), (3));
            assert_eq!(idx.find_block(&Timestamp(10)), (3));
            assert_eq!(idx.find_block(&Timestamp(11)), (5));
        }
    }
}

pub mod clock {
    use std::time::{Duration, SystemTime};
    use crate::engine::Timestamp;

    /// Source of time that provides timestamps that do not go backwards in time.
    pub struct Clock {
        last_timestamp: Duration,
    }

    impl Clock {
        pub fn now(&mut self) -> Timestamp {
            self.last_timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(self.last_timestamp);
            Timestamp(self.last_timestamp.as_secs())
        }
    }

    impl Default for Clock {
        fn default() -> Self {
            Clock {
                last_timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("cannot initialize initial_timestamp")
            }
        }
    }
}

pub mod server {
    use log::debug;
    use crate::engine::{Schema, Timestamp, Point};
    use crate::engine::index::TimestampIndex;
    use crate::engine::block::{BlockSpec, BLOCK_TAIL_SIZE, BLOCK_SIZE, Block};
    use crate::engine::cache::Cache;
    use crate::engine::io::{BlockIO, SyncPolicy};
    use crate::engine::clock::Clock;
    use crate::engine::Decoder;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Type that contains metadata about series.
    pub struct Series<S, V> where S: Schema<V> {
        pub(crate) id: usize,
        pub(crate) enc_state: S::EncState,
        pub(crate) timestamp_index: TimestampIndex,
        pub(crate) blocks: usize,
        pub(crate) last_block_used_bytes: usize,
        pub(crate) last_timestamp: Timestamp,
    }

    impl<S, V> Series<S, V> where S: Schema<V> {
        #[inline]
        fn create_block_spec(&self, block_id: usize) -> BlockSpec {
            BlockSpec { series_id: self.id, block_id }
        }

        /// Returns new BlockSpec referencing current last Block
        /// in this Series structure.
        pub fn last_block_spec(&self) -> BlockSpec {
            self.create_block_spec(self.blocks - 1)
        }

        #[inline]
        pub fn last_block_free_bytes(&self) -> usize {
            (BLOCK_SIZE - BLOCK_TAIL_SIZE) - self.last_block_used_bytes
        }
    }

    pub struct BlockLoader {
        cache: Cache<BlockSpec, Block>,
        io: BlockIO,
    }

    impl BlockLoader {
        /// Creates new instance of Block loader with specified
        /// storage path, block cache capacity and sync policy for writing.
        pub fn new(storage: PathBuf, cache_capacity: usize, sync_policy: SyncPolicy) -> Self {
            if !storage.exists() {
                debug!("Storage dir does not exists - creating... {}", storage.to_string_lossy());
                std::fs::create_dir_all(&storage)
                    .expect("cannot create storage directory");
            }

            debug!("Creating BlockLoader storage={} cache_capacity={} sync_policy={:?}",
                   storage.to_string_lossy(), cache_capacity, sync_policy);
            BlockLoader {
                cache: Cache::with_capacity(cache_capacity),
                io: BlockIO::new(storage, sync_policy),
            }
        }

        /// Acquires block either from cache or by loading from disk and
        /// returns it as immutable reference.
        pub fn acquire_block(&mut self, spec: BlockSpec) -> &Block {
            if !self.cache.contains_key(&spec) {
                self.cache.insert(spec, self.io.create_or_load_block(&spec));
            }

            self.cache.get_mut(&spec)
                .expect("block is not in cache after we loaded it!")
        }

        /// Removes the Block specified by `spec` parameter from the cache
        /// if it is present. Calling this method will remove Block from
        /// cache **even if it is unevictable**.
        pub fn remove(&mut self, spec: &BlockSpec) {
            self.cache.remove(spec);
        }

        /// Acquires mutable reference to Block specified by `spec` parameter,
        /// calls the specified closure with it and then writes the Block
        /// to disk partially using `dirty_from` parameter.
        pub fn acquire_then_write<F: (FnOnce(&mut Block) -> ())>(&mut self,
                                                                 spec: BlockSpec,
                                                                 dirty_from: usize,
                                                                 f: F) {
            if !self.cache.contains_key(&spec) {
                self.cache.insert(spec, self.io.create_or_load_block(&spec));
            }

            let b = self.cache
                .get_mut(&spec)
                .expect("block is not in cache after we loaded it!");
            f(b);
            self.io.write_block_partially(&spec, b, dirty_from);
        }

        /// Acquires mutable reference to Block specified by `spec` parameter,
        /// calls the specified closure with it and then writes the Block
        /// to disk partially using `dirty_from` parameter.
        ///
        /// This method has same effect as calling `acquire_then_write` however
        /// the block is loaded as unevictable meaning that it will not be
        /// evicted when another Block is loaded into the cache.
        pub fn acquire_unevictable_then_write<F: (FnOnce(&mut Block) -> ())>(&mut self,
                                                                             spec: BlockSpec,
                                                                             dirty_from: usize,
                                                                             f: F) {
            if !self.cache.contains_key(&spec) {
                self.cache.insert_unevictable(spec, self.io.create_or_load_block(&spec));
            }

            let b = self.cache
                .get_mut(&spec)
                .expect("block is not in cache after we loaded it!");
            f(b);
            self.io.write_block_partially(&spec, b, dirty_from);
        }

        /// Acquires mutable reference to Block specified by `spec` parameter,
        /// calls the specified closure with it.
        pub fn acquire_unevictable<F: (FnOnce(&mut Block) -> ())>(&mut self,
                                                                  spec: BlockSpec,
                                                                  f: F) {
            if !self.cache.contains_key(&spec) {
                self.cache.insert_unevictable(spec, self.io.create_or_load_block(&spec));
            }

            let b = self.cache
                .get_mut(&spec)
                .expect("block is not in cache after we loaded it!");
            f(b);
        }
    }

    /// Type that represents fully-functional storage system for
    /// time-series data.
    pub struct Engine<S, V> where S: Schema<V> {
        pub(crate) clock: Clock,
        pub(crate) series: HashMap<String, Series<S, V>>,
        pub(crate) last_series_id: usize,
        pub(crate) storage: PathBuf,
        pub(crate) sync_policy: SyncPolicy,
        pub(crate) blocks: BlockLoader,
    }

    impl<S, V> Engine<S, V> where S: Schema<V> {
        pub fn new(storage: PathBuf, cache_capacity: usize, sync_policy: SyncPolicy) -> Self {
            Engine {
                clock: Default::default(),
                series: Default::default(),
                last_series_id: 0,
                storage: storage.clone(),
                sync_policy,
                blocks: BlockLoader::new(storage, cache_capacity, sync_policy),
            }
        }

        fn insert_point_at(&mut self, series_name: &str, value: V, timestamp: Timestamp) {
            let mut series = self.series
                .get_mut(series_name)
                .unwrap();

            let point = Point { timestamp, value };
            let encoded = S::encode(&mut series.enc_state, point);
            {
                let last_block_free_bytes = series.last_block_free_bytes();

                if last_block_free_bytes < encoded.length {
                    let last_block_spec = series.last_block_spec();

                    // load the block if for some reason is not loaded and write the free bytes
                    self.blocks.acquire_then_write(last_block_spec, series.last_block_used_bytes, |b| {
                        b.set_free_bytes(last_block_free_bytes as u8);
                    });

                    // as it might be loaded as unevictable we have to manually remove it from cache
                    self.blocks.remove(&last_block_spec);

                    series.timestamp_index.set_max(last_block_spec.block_id, series.last_timestamp);
                    series.timestamp_index.set_min(last_block_spec.block_id + 1, timestamp);
                    series.timestamp_index.write_dirty_part();
                    series.blocks += 1;
                    series.last_block_used_bytes = 0;
                }
            }

            // closure that writes the encoded data into block
            let fn_write_encoded = |b: &mut Block| {
                for (i, x) in encoded.data[0..encoded.length].iter().enumerate() {
                    b.data[series.last_block_used_bytes + i] = *x;
                }
            };

            let block_spec = series.last_block_spec();
            self.blocks.acquire_unevictable_then_write(
                block_spec,
                series.last_block_used_bytes,
                fn_write_encoded,
            );
            series.timestamp_index.set_max(block_spec.block_id, timestamp);
            series.timestamp_index.write_dirty_part();
            series.last_timestamp = timestamp;
            series.last_block_used_bytes += encoded.length;
        }
    }

    pub enum CreateSeriesError {
        SeriesAlreadyExists
    }

    pub enum InsertPointError {
        SeriesNotExists
    }

    pub enum RetrievePointsError {
        SeriesNotExists
    }

    pub trait SimpleServer<V> {
        fn create_series(&mut self, name: &str) -> Result<(), CreateSeriesError>;
        fn insert_point(&mut self, series_name: &str, value: V) -> Result<(), InsertPointError>;
        fn retrieve_points(&mut self, series_name: &str, from: Option<Timestamp>, to: Option<Timestamp>) -> Result<Vec<Point<V>>, RetrievePointsError>;
    }

    impl<S, V> SimpleServer<V> for Engine<S, V> where S: Schema<V> {
        /// Creates a new series object.
        fn create_series(&mut self, name: &str) -> Result<(), CreateSeriesError> {
            if self.series.contains_key(name) {
                return Err(CreateSeriesError::SeriesAlreadyExists);
            }

            let series_id = self.last_series_id;
            let index_file = self.storage.join(format!("{}.idx", series_id));
            let index = TimestampIndex::create_or_load(&index_file, self.sync_policy)
                .unwrap();

            debug!("Creating new series ID {} ('{}')", series_id, name);
            self.series.insert(name.to_string(), Series {
                id: series_id,
                enc_state: Default::default(),
                timestamp_index: index,
                blocks: 1,
                last_block_used_bytes: 0,
                last_timestamp: Timestamp(0),
            });
            self.last_series_id += 1;
            Ok(())
        }

        /// Inserts specified point into the specified Series object.
        fn insert_point(&mut self, series_name: &str, value: V) -> Result<(), InsertPointError> {
            if !self.series.contains_key(series_name) {
                return Err(InsertPointError::SeriesNotExists);
            }

            let timestamp = self.clock.now();
            Ok(self.insert_point_at(series_name, value, timestamp))
        }

        /// Retrieves all data points that happened between `from` and `to`
        /// parameters. If the parameters are empty (`None`) the range is
        /// considered open from one or both sides.
        fn retrieve_points(&mut self,
                           series_name: &str,
                           from: Option<Timestamp>,
                           to: Option<Timestamp>) -> Result<Vec<Point<V>>, RetrievePointsError> {
            if !self.series.contains_key(series_name) {
                return Err(RetrievePointsError::SeriesNotExists);
            }

            let mut points = vec![];
            let series = self.series
                .get_mut(series_name)
                .unwrap();

            let start_block = from
                .or_else(|| Some(Timestamp(0)))
                .map(|x| series.timestamp_index.find_block(&x))
                .unwrap();

            // we start at block that does not exists yet -> empty result
            if start_block > series.blocks - 1 {
                return Ok(vec![]);
            }

            let end_block = to
                .map(|x| series.timestamp_index.find_block(&x))
                .map(|x| x.min(series.blocks - 1))
                .unwrap_or(series.blocks - 1);

            // first timestamp in block 0 is absolute and not relative
            let min_timestamp = if start_block == 0 { Timestamp(0) } else {
                *series.timestamp_index.get_min(start_block)
            };

            let mut dec_state = S::DecState::new(min_timestamp);

            for block_id in start_block..=end_block {
                let spec = series.create_block_spec(block_id);
                let block = self.blocks.acquire_block(spec);

                let mut read_bytes = 0;
                let written_bytes = series.last_block_used_bytes;
                while read_bytes < written_bytes {
                    let offset_buff = &block.data[read_bytes..];
                    let (point, rb) = S::decode(&mut dec_state, offset_buff);

                    if to.is_some() && to.unwrap().0 < point.timestamp.0 {
                        break;
                    }

                    if from.is_none() || from.unwrap().0 <= point.timestamp.0 {
                        points.push(point);
                    }

                    read_bytes += rb
                }
            }

            Ok(points)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::engine::server::{Engine, SimpleServer};
        use crate::engine::f32::F32;
        use crate::engine::io::SyncPolicy;
        use tempdir::TempDir;
        use crate::engine::Timestamp;

        #[test]
        fn test_simple_commands() {
            let dir = TempDir::new("test_simple_commands").unwrap();
            let mut s: Engine<F32, f32> = Engine::new(
                dir.path().join("storage/"),
                1024,
                SyncPolicy::Never,
            );

            s.create_series("default");
            s.insert_point_at("default", 1.0, Timestamp(1));
            s.insert_point_at("default", 2.0, Timestamp(1));
            s.insert_point_at("default", 3.0, Timestamp(2));
            s.insert_point_at("default", 4.0, Timestamp(3));
            s.insert_point_at("default", 5.0, Timestamp(4));
            s.insert_point_at("default", 6.0, Timestamp(5));
            s.insert_point_at("default", 7.0, Timestamp(5));
            s.insert_point_at("default", 8.0, Timestamp(5));
            s.insert_point_at("default", 9.0, Timestamp(10));
            s.insert_point_at("default", 10.0, Timestamp(11));

            assert_eq!(s.retrieve_points("default", None, None).ok().unwrap().len(), 10);
            assert_eq!(s.retrieve_points("default", Some(Timestamp(500)), None).ok().unwrap().len(), 0);
            assert_eq!(s.retrieve_points("default", Some(Timestamp(1)), Some(Timestamp(11))).ok().unwrap().len(), 10);
            assert_eq!(s.retrieve_points("default", Some(Timestamp(1)), Some(Timestamp(500))).ok().unwrap().len(), 10);
            assert_eq!(s.retrieve_points("default", Some(Timestamp(1)), Some(Timestamp(10))).ok().unwrap().len(), 9);
            assert_eq!(s.retrieve_points("default", Some(Timestamp(3)), Some(Timestamp(10))).ok().unwrap().len(), 6);
            assert_eq!(s.retrieve_points("default", Some(Timestamp(5)), Some(Timestamp(5))).ok().unwrap().len(), 3);
            assert_eq!(s.retrieve_points("default", Some(Timestamp(10)), Some(Timestamp(1))).ok().unwrap().len(), 0);
            assert_eq!(s.retrieve_points("default", Some(Timestamp(11)), Some(Timestamp(100))).ok().unwrap().len(), 1);
        }
    }
}
