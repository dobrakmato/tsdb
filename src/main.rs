
use std::collections::HashMap;
use std::path::PathBuf;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Read};

const BLOCK_SIZE: usize = 4096;
const BLOCKS_PER_FILE: usize = 2048;

/// Simple new-type struct representing the block.
struct Block([u8; BLOCK_SIZE]);

/*************************/

/// Enumeration of all supported schemas
enum Schema {
    F32
}

impl Schema {

    // make this const fn when this is fixed: https://github.com/rust-lang/rust/issues/49146
    fn datum_size(&self) -> usize {
        match self {
            Schema::F32 => std::mem::size_of::<f32>(),
        }
    }
}

/// datas about datas in this series
struct Metadata {
    /* how sparse the data is */
    sparsity: f32,
    /* avg delta between consecutive data points in seconds */
    avg_timestamp_delta: f32,
}

/// information about blocks in this series
struct BlockInfo {
    block_size: usize,
    /* number of existing blocks. also current (last) block. */
    block_count: usize,
    /* number of used bytes inside the last block */
    last_block_used_bytes: usize,
}

struct Series {
    name: String,
    schema: Schema,
    metadata: Metadata,
    block_info: BlockInfo,
    index: Index,
}

struct Index {
    timestamp_start: Vec<usize>,
    timestamp_end: Vec<usize>,
}

/******************************/

/// contains currently loaded blocks
struct BlockCache {
    loaded_blocks: Vec<Block>
}

/// contains all unprocessed operations
struct RedoLog {}

/// contains logic for loading block data from disk
struct BlockLoader {
    storage: PathBuf,
    open_files: HashMap<PathBuf, File>,
}

impl BlockLoader {
    pub fn load_block(&self, series_name: &str, block_id: usize) -> Block {
        // Determine file id and then file path.
        let file_id = block_id / BLOCKS_PER_FILE;
        let file_path = self.storage.join(format!("{}-{}.dat", series_name, file_id));

        // Check if the current file is currently open. If not
        // open the file. Then seek to correct block and read it.
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .open(file_path)
            .expect("cannot create or open file");

        let local_block_id = block_id - (file_id * BLOCKS_PER_FILE);
        let block_start = local_block_id * BLOCK_SIZE;
        file.seek(SeekFrom::Start(block_start as u64)).unwrap();

        let mut block = Block([0; BLOCK_SIZE]);
        file.read_exact(&mut block.0);
        return block;
    }
}

struct Server {
    series: HashMap<String, Series>,
    block_cache: BlockCache,
    block_loader: BlockLoader,
    redo_log: RedoLog,
}

fn main() {
    println!("size of block {}", std::mem::size_of::<Block>());
    println!("align of block {}", std::mem::align_of::<Block>());
}
