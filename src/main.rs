#[macro_use]
extern crate log;

use std::collections::HashMap;
use std::path::{PathBuf, Path};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Read, Write, Error, ErrorKind};
use std::collections::hash_map::Entry;
use std::time::{SystemTime, Duration, Instant};

const BLOCK_SIZE: usize = 4096;
const BLOCKS_PER_FILE: usize = 2048;
const ENTRY_MAX_SIZE: usize = 16;

/// Simple new-type struct representing the block.
struct Block([u8; BLOCK_SIZE]);

/// Simple small-vec type used to represent BlockEntry.
struct BlockEntry([u8; ENTRY_MAX_SIZE], usize);

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

    fn encode(&self, last_timestamp: u64, current_timestamp: u64, last_value: f32, current_value: f32) -> BlockEntry {
        let mut entry_buffer = [0u8; ENTRY_MAX_SIZE];
        let mut entry_size = 0;

        match self {
            Schema::F32 => {
                let dt = current_timestamp - last_timestamp;
                let dv = current_value - last_value;

                let mut writable: &mut [u8] = &mut entry_buffer;

                entry_size += leb128::write::unsigned(&mut writable, dt).unwrap();

                dv.to_le_bytes()
                    .iter()
                    .enumerate()
                    .for_each(|(i, x)| entry_buffer[entry_size + i] = *x);
                entry_size += 4;
            }
        }

        return BlockEntry(entry_buffer, entry_size);
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
    last_block_last_timestamp: u64,
    last_block_last_value: f32,
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
}

#[derive(Debug)]
struct Index {
    timestamp_start: Vec<usize>,
    timestamp_end: Vec<usize>,
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

        let mut block = Block([0; BLOCK_SIZE]);

        // The file can either have contents (in which case we just read
        // it) or the block is not yet allocated (for example if we just
        // created the file and are writing first block).
        file.read_exact(&mut block.0).unwrap();
        return block;
    }

    /// Writes block data into file specified by BlockSpec with data
    /// that is stored in provided Block.
    pub fn write_and_flush_block(&mut self, block_spec: &BlockSpec, block: &Block) {
        let file = self.load_or_create_file_for_block(block_spec);
        BlockIO::seek_to_block_start(file, block_spec);

        file.write_all(&block.0).unwrap();
        // file.sync_all().unwrap();
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
    fn create_series(&mut self) {}

    pub fn insert(&mut self, series: &str, value: f32) {
        let mut series = self.series.get_mut(series).unwrap();

        // compute the data we are going to write to some block
        let current_timestamp = self.time_source.get_timestamp();
        let encoded = series.schema.encode(
            series.blocks_info.last_block_last_timestamp,
            current_timestamp,
            series.blocks_info.last_block_last_value,
            value,
        );
        series.blocks_info.last_block_last_value = value;
        series.blocks_info.last_block_last_timestamp = current_timestamp;

        // check if the block is full and we need to create new block
        if BLOCK_SIZE - series.blocks_info.last_block_used_bytes < encoded.1 {
            series.blocks_info.block_count += 1;
            series.blocks_info.last_block_used_bytes = 0;
        }

        let last_block_spec = series.create_last_block_spec();
        let last_block = match self.block_cache.get_mut(&last_block_spec) {
            Some(t) => t,
            None => {
                let block = self.block_io.load_or_create_block(&last_block_spec);
                self.block_cache.insert(last_block_spec.clone(), block)
            }
        };

        // here we actually write the timestamp and provided value to the
        // block. we need to trim the buffer or zeros at the end.
        let actual_encoded_data = &encoded.0[0..encoded.1];
        for (i, x) in actual_encoded_data.iter().enumerate() {
            last_block.0[series.blocks_info.last_block_used_bytes + i] = *x;
        }
        series.blocks_info.last_block_used_bytes += actual_encoded_data.len();

        // now just write the block and flush
        self.block_io.write_and_flush_block(&last_block_spec, last_block);
    }
}

fn main() {
    simple_logger::init().unwrap();

    let mut series = Series {
        name: "default".to_owned(),
        schema: Schema::F32,
        metadata: Metadata { sparsity: 0.0, avg_timestamp_delta: 0.0 },
        blocks_info: BlocksInfo {
            block_size: BLOCK_SIZE,
            block_count: 1,
            last_block_used_bytes: 0,
            last_block_last_timestamp: 0,
            last_block_last_value: 0.0,
        },
        index: Index { timestamp_start: vec![], timestamp_end: vec![] },
    };

    let mut server = Server {
        series: Default::default(),
        block_cache: BlockCache::new(1024), // 4mb
        block_io: BlockIO {
            storage: PathBuf::from("./storage"),
            open_files: Default::default(),
        },
        time_source: TimeSource::new(),
    };

    server.series.insert("default".to_owned(), series);

    let records = 50000;

    let start = Instant::now();
    for i in 0..records {
        server.insert("default", i as f32);
    }
    println!("inserted {} records in {}s", records, start.elapsed().as_secs_f32());
    println!("{:#?}", server.series.get("default"));
}
