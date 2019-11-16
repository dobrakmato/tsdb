#[macro_use]
extern crate log;

use std::collections::HashMap;
use std::path::{PathBuf, Path};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Read, Write};
use std::collections::hash_map::Entry;
use std::time::{SystemTime, Duration, Instant};
use static_assertions::{assert_eq_align, assert_eq_size};
use std::fmt::{Debug, Formatter, Error};
use nano_leb128::ULEB128;

const BLOCK_SIZE: usize = 4096;
const BLOCKS_PER_FILE: usize = 2048;

/// Header of each block.
struct BlockHeader {
    free_bytes: u8,
}

const BLOCK_HEADER_SIZE: usize = std::mem::size_of::<BlockHeader>();

/// Simple new-type struct representing the block.
struct Block(BlockHeader, [u8; BLOCK_SIZE - BLOCK_HEADER_SIZE]);

// These assertions ensure we can safely interpret Block structure as [u8; 4096]
assert_eq_size!(Block, [u8; BLOCK_SIZE]);
assert_eq_align!(Block, u8);

struct SmallVec {
    storage: [u8; Self::MAX_SIZE],
    len: u8,
}

impl SmallVec {
    const MAX_SIZE: usize = 16;

    pub fn new() -> Self {
        SmallVec { storage: [0; 16], len: 0 }
    }

    pub fn data(&self) -> &[u8] {
        return &self.storage[0..self.len as usize];
    }

    pub fn write(&mut self, value: u8) {
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
/*************************/

/// Enumeration of all supported schemas
#[derive(Debug)]
enum Schema {
    F32
}

impl Schema {
    // todo: make this const fn when this is fixed: https://github.com/rust-lang/rust/issues/49146
    fn datum_size(&self) -> usize {
        match self {
            Schema::F32 => std::mem::size_of::<f32>(),
        }
    }

    fn codec(&self) -> impl Codec {
        match self {
            Schema::F32 => F32 {},
        }
    }
}

trait Codec {
    type EncoderState: Default;
    type DecoderState: Default;
    fn encode(current_timestamp: u64, current_value: f32, state: &mut Self::EncoderState) -> SmallVec;
    fn decode(buff: &[u8], state: &mut Self::DecoderState) -> (u64, f32, usize);
}

#[derive(Default, Debug)]
struct F32State {
    last_timestamp: u64,
}

struct F32;

/// Timestamp: LEB128 delta-encoded
/// Value: F32 uncompressed
impl Codec for F32 {
    type EncoderState = F32State;
    type DecoderState = F32State;

    fn encode(current_timestamp: u64, current_value: f32, state: &mut Self::EncoderState) -> SmallVec {
        let mut vec = SmallVec::new();

        let dt = current_timestamp - state.last_timestamp;
        vec.len += ULEB128::from(dt).write_into(&mut vec.storage).unwrap() as u8;

        current_value.to_le_bytes()
            .iter()
            .for_each(|x| vec.write(*x));

        state.last_timestamp = current_timestamp;

        return vec;
    }

    fn decode(buff: &[u8], state: &mut Self::DecoderState) -> (u64, f32, usize) {
        let (dt, read) = ULEB128::read_from(buff).unwrap();
        let value = f32::from_le_bytes([
            buff[read + 0],
            buff[read + 1],
            buff[read + 2],
            buff[read + 3]
        ]);

        let ct = state.last_timestamp + u64::from(dt);
        state.last_timestamp = ct;

        return (ct, value, read + 4);
    }
}

/// datas about datas in this series
#[derive(Debug)]
struct Metadata {
    /* how sparse the data is */
    sparsity: f32,
    /* avg delta between consecutive data points in seconds */
    avg_timestamp_delta: f32,
}

/// information about blocks in this series
#[derive(Debug)]
struct BlocksInfo {
    block_size: usize,
    /* number of existing blocks. also current (last) block. */
    block_count: usize,
    /* number of used bytes inside the last block */
    last_block_used_bytes: usize,
    encoder_state: F32State,
}

#[derive(Debug)]
struct Series {
    name: String,
    schema: Schema,
    metadata: Metadata,
    blocks_info: BlocksInfo,
    index: Index,
}

impl Series {
    fn create_block_spec(&self, block_id: usize) -> BlockSpec {
        BlockSpec { series_name: self.name.clone(), block_id }
    }

    /// Creates a BlockSpec representing the last block in this
    /// series object (the one we are going to write to).
    pub fn create_last_block_spec(&self) -> BlockSpec {
        return self.create_block_spec(self.blocks_info.block_count - 1);
    }

    pub fn is_last_block(&self, block_id: usize) -> bool {
        return self.blocks_info.block_count - 1 == block_id;
    }
}

struct Index {
    timestamp_min: Vec<u64>,
    timestamp_max: Vec<u64>,
}

impl Index {
    fn ensure_vec_has_enough_storage(vec: &mut Vec<u64>, for_block_id: usize) {
        assert!(vec.len() >= for_block_id);

        if vec.len() < for_block_id + 1 {
            // when we push to the full Vec in Rust the backing
            // allocation size is doubled. this is not really what we
            // want because our Index grows linearly and predictably.
            // for that reason we reserve additional 8 bytes manually.
            if vec.capacity() == vec.len() {
                vec.reserve(8);
            }
            vec.push(0);
        }
    }

    /// block_id must be valid block id
    pub fn set_min(&mut self, block_id: usize, timestamp: u64) {
        Index::ensure_vec_has_enough_storage(&mut self.timestamp_min, block_id);
        self.timestamp_min[block_id] = timestamp;
    }

    pub fn set_max(&mut self, block_id: usize, timestamp: u64) {
        Index::ensure_vec_has_enough_storage(&mut self.timestamp_max, block_id);
        self.timestamp_max[block_id] = timestamp;
    }

    pub fn min_for(&self, block_id: usize) -> u64 {
        return self.timestamp_min[block_id];
    }
}

impl Debug for Index {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_fmt(format_args!("Index {{ min_len={}, max_len={} }}", self.timestamp_min.len(), self.timestamp_max.len()))
    }
}

/******************************/

/// A "handle" that uniquely represents a Block inside the application.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct BlockSpec {
    series_name: String,
    block_id: usize,
}

impl BlockSpec {
    /// Creates a PathBuf containing relative path to file this
    /// block should reside in.
    pub fn determine_file_path(&self, base_path: &Path) -> PathBuf {
        let file_id = self.block_id / BLOCKS_PER_FILE;
        return base_path.join(format!("{}-{}.dat", self.series_name, file_id));
    }

    /// Computes position in bytes this block starts within the
    /// file this block resides in.
    pub fn starting_pos_in_file(&self) -> usize {
        let file_id = self.block_id / BLOCKS_PER_FILE;
        let file_local_block_id = self.block_id - (file_id * BLOCKS_PER_FILE);
        return file_local_block_id * BLOCK_SIZE;
    }
}

/// LRU cache that contains currently loaded Blocks.
struct BlockCache {
    loaded_blocks: HashMap<BlockSpec, Block>,
    lru: Vec<Option<BlockSpec>>,
    idx: usize,
}

impl BlockCache {
    /// Creates a new instance of BlockCache with specified number
    /// of maximum elements. Elements are evicted from the cache in
    /// least recently used fashion.
    pub fn new(max_size: usize) -> Self {
        BlockCache {
            loaded_blocks: HashMap::with_capacity(max_size),
            lru: vec![None; max_size],
            idx: 0,
        }
    }

    fn evict_current_if_present(&mut self) {

        //todo: do not evict LAST blocks of series from cache

        if let Some(t) = &self.lru[self.idx] {
            debug!("evicted block {} ({}) from cache", t.block_id, t.series_name);
            self.loaded_blocks.remove(t);
            self.lru[self.idx] = None;
        }
    }

    pub fn insert(&mut self, block_spec: BlockSpec, block: Block) -> &mut Block {
        self.evict_current_if_present();
        self.lru[self.idx] = Some(block_spec.clone());
        self.loaded_blocks.insert(block_spec, block);

        // here we create mutable reference for the block we just
        // inserted into the hashmap because caller probably wants
        // to use the block anyways.
        let mut_ref = self.loaded_blocks.get_mut(&self.lru[self.idx].as_ref().unwrap()).unwrap();

        // we wrap around the vec as if it was a ring buffer
        self.idx = (self.idx + 1) % self.lru.len();

        return mut_ref;
    }

    pub fn contains_block(&self, block_spec: &BlockSpec) -> bool {
        return self.loaded_blocks.contains_key(block_spec);
    }

    /// Returns a block specified by BlockSpec from the cache if it
    /// is in the cache, otherwise returns None.
    pub fn get_mut(&mut self, block_spec: &BlockSpec) -> Option<&mut Block> {
        return self.loaded_blocks.get_mut(&block_spec);
    }
}


/// contains logic for loading & saving block data from disk
struct BlockIO {
    storage: PathBuf,
    open_files: HashMap<PathBuf, File>,
}

impl BlockIO {
    /// Loads (or creates) file needed to work with block specified
    /// by series name and it's ID inside the series. This method
    /// either returns already opened file, or opens a new one.
    fn load_or_create_file_for_block(&mut self, block_spec: &BlockSpec) -> &mut File {
        // Determine file id and then file path.
        let file_path = block_spec.determine_file_path(&self.storage);

        // Check if the current file is currently open. If not
        // open (and create if necessary) the file
        return match self.open_files.entry(file_path) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // here we need to clone file_path because it will be moved
                // into `entry.insert()`.
                let file_path = entry.key().clone();

                let mut f = OpenOptions::new()
                    .write(true)
                    .read(true)
                    .create(true)
                    .open(file_path)
                    .expect("cannot create or open file");

                // pre-allocate file content.
                f.write_all(&[0u8; BLOCKS_PER_FILE * BLOCK_SIZE]).expect("cannot pre-allocate file");
                f.sync_all().expect("cannot sync pre-allocated data to disk");

                entry.insert(f)
            }
        };
    }

    fn seek_to_block_start(file: &mut File, block_spec: &BlockSpec) {
        // Compute starting position of requested block and seek the open
        // file to specified position and read block form the file.
        let seek = SeekFrom::Start(block_spec.starting_pos_in_file() as u64);
        file.seek(seek).expect("cannot seek to specified position!");
    }

    /// Loads block data for block specified by passed BlockSpec
    /// from file and returns new instance of Block containing
    /// requested data.
    pub fn load_or_create_block(&mut self, block_spec: &BlockSpec) -> Block {
        let file = self.load_or_create_file_for_block(block_spec);
        BlockIO::seek_to_block_start(file, block_spec);

        let header = BlockHeader { free_bytes: 0 };
        let mut block = Block(header, [0; BLOCK_SIZE - BLOCK_HEADER_SIZE]);

        // Here we need to load the data from file into memory
        // to prevent copying we will interpret the new Block structure
        // instance as if it was a 4096 bytes long u8 array.
        //
        // To do this safely we must ensure the proper size and
        // alignment. For this reason we have static assertions.
        //
        // Safe: because Block has size of BLOCK_SIZE and alignment of 1.
        let ptr = &mut block as *mut Block as *mut u8;
        let mut bytes = unsafe {
            std::slice::from_raw_parts_mut(ptr, BLOCK_SIZE)
        };

        file.read_exact(&mut bytes).unwrap();
        return block;
    }

    /// Writes block data into file specified by BlockSpec with data
    /// that is stored in provided Block.
    pub fn write_and_flush_block(&mut self, block_spec: &BlockSpec, block: &Block) {
        let file = self.load_or_create_file_for_block(block_spec);
        BlockIO::seek_to_block_start(file, block_spec);

        // Here we need to write data of the block into the
        // file. We can prevent copying by interpreting the Block
        // structure as if it was 4096 bytes long array.
        //
        // Safe: because Block has size of BLOCK_SIZE and alignment of 1.
        let ptr = block as *const Block as *const u8;
        let bytes = unsafe {
            std::slice::from_raw_parts(ptr, BLOCK_SIZE)
        };

        file.write_all(bytes).unwrap();
        // file.sync_all().unwrap(); todo: fsync policy
    }
}

struct TimeSource {
    last_timestamp: Duration,
}

impl TimeSource {
    pub fn new() -> Self {
        TimeSource {
            last_timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("cannot initialize initial_timestamp")
        }
    }

    pub fn get_timestamp(&mut self) -> u64 {
        self.last_timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(self.last_timestamp);
        return self.last_timestamp.as_secs();
    }
}

struct Server {
    series: HashMap<String, Series>,
    block_cache: BlockCache,
    block_io: BlockIO,
    time_source: TimeSource,
}

impl Server {
    fn acquire_block<'a>(block_cache: &'a mut BlockCache, block_io: &mut BlockIO, block_spec: &BlockSpec) -> &'a mut Block {
        if block_cache.contains_block(&block_spec) {
            return block_cache.get_mut(&block_spec).unwrap();
        }
        let block = block_io.load_or_create_block(&block_spec);
        block_cache.insert(block_spec.clone(), block)
    }

    fn create_series(&mut self, name: &str, schema: Schema) {
        let series = Series {
            name: name.to_owned(),
            schema,
            metadata: Metadata { sparsity: 0.0, avg_timestamp_delta: 0.0 },
            blocks_info: BlocksInfo {
                block_size: BLOCK_SIZE,
                block_count: 1,
                last_block_used_bytes: 0,
                encoder_state: Default::default(),
            },
            index: Index { timestamp_min: vec![], timestamp_max: vec![] },
        };
        self.series.insert(name.to_owned(), series);
    }

    pub fn insert(&mut self, series: &str, value: f32) {
        let mut series = self.series.get_mut(series).unwrap();

        // compute the data we are going to write to some block
        let current_timestamp = self.time_source.get_timestamp();
        let encoded = F32::encode(current_timestamp, value, &mut series.blocks_info.encoder_state);

        // decide and return the block we should write data to.
        let (block_spec, mut block) = {
            let mut block_spec = series.create_last_block_spec();
            let mut block = Server::acquire_block(&mut self.block_cache, &mut self.block_io, &block_spec);

            // check if the block is full and we need to create new block
            if BLOCK_SIZE - series.blocks_info.last_block_used_bytes < encoded.len() {
                // to properly close current block we should write the number
                // of unused bytes to the block's start so we can later decode
                // the data from it.
                let last_block_free_bytes = block.1.len() - series.blocks_info.last_block_used_bytes;
                block.0.free_bytes = last_block_free_bytes as u8;
                self.block_io.write_and_flush_block(&block_spec, block);

                // index: as this is the last value in this block we should
                // update timestamp_max index.
                series.index.set_max(block_spec.block_id, current_timestamp);

                // we can now proceed with creating a new block
                series.blocks_info.block_count += 1;
                series.blocks_info.last_block_used_bytes = 0;

                block_spec = series.create_last_block_spec();
                block = Server::acquire_block(&mut self.block_cache, &mut self.block_io, &block_spec);
            }

            (block_spec, block)
        };

        // index: if this is the first value in this block we should
        // update timestamp_min index.
        if series.blocks_info.last_block_used_bytes == 0 {
            series.index.set_min(block_spec.block_id, current_timestamp);
        }

        // here we actually write the timestamp and provided value to the
        // block. we need to trim the buffer or zeros at the end.
        let actual_encoded_data = encoded.data();
        for (i, x) in actual_encoded_data.iter().enumerate() {
            block.1[series.blocks_info.last_block_used_bytes + i] = *x;
        }
        series.blocks_info.last_block_used_bytes += actual_encoded_data.len();

        // now just write the block and flush
        self.block_io.write_and_flush_block(&block_spec, block);
    }

    pub fn select_avg(&mut self, series: &str) -> f32 {
        let mut agg = AvgAgg::new();
        let mut series = self.series.get_mut(series).unwrap();

        // find indices using binary search in index
        let start_block = 0;
        let end_block = series.blocks_info.block_count - 1;

        // first block in series has absolute timestamp, but search can
        // start at other than first block so we must lookup last_timestamp
        // using index in that case.
        let min_timestamp = if start_block == 0 { 0 } else { series.index.min_for(start_block) };
        let mut decoder_state = F32State { last_timestamp: min_timestamp };

        // linear scan trough blocks
        for block_id in start_block..=end_block {
            let spec = series.create_block_spec(block_id);
            let block = Server::acquire_block(&mut self.block_cache, &mut self.block_io, &spec);

            // if end_block is last block of the series we should use
            // information about values from BlockInfo rather from block itself
            // as the block contains zero as number of free_bytes.
            let block_used_bytes = if series.is_last_block(block_id) {
                series.blocks_info.last_block_used_bytes
            } else {
                block.1.len() - block.0.free_bytes as usize
            };

            let mut i = 0;
            while i < block_used_bytes {
                let offset_buff = &block.1[i..];
                let (ts, value, read) = F32::decode(offset_buff, &mut decoder_state);

                agg.next(value);

                i += read;
            }
        }

        return agg.avg;
    }
}

struct AvgAgg {
    avg: f32,
    t: u64,
}

impl AvgAgg {
    pub fn new() -> Self {
        AvgAgg { avg: 0.0, t: 1 }
    }

    pub fn next(&mut self, item: f32) {
        self.avg += (item - self.avg) / self.t as f32;
        self.t += 1;
    }
}

fn main() {
    simple_logger::init().unwrap();

    let mut server = Server {
        series: Default::default(),
        block_cache: BlockCache::new(1024), // 4mb
        block_io: BlockIO {
            storage: PathBuf::from("./storage"),
            open_files: Default::default(),
        },
        time_source: TimeSource::new(),
    };

    server.create_series("default", Schema::F32);

    let records = 50000;

    let start = Instant::now();
    for i in 0..records {
        server.insert("default", i as f32);
    }
    println!("inserted {} records in {}s", records, start.elapsed().as_secs_f32());
    println!("{:#?}", server.series.get("default").unwrap());

    let start = Instant::now();
    let avg = server.select_avg("default");
    println!("computed avg={} in {}s", avg, start.elapsed().as_secs_f32());
}
