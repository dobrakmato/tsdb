#![feature(float_to_from_bytes)]

use crate::block::Block;
use std::iter::Map;
use std::collections::HashMap;

mod block;

enum Schema {
    F32
}

struct Metadata {
    /* how sparse the data is */
    sparsity: f32,
    /* avg delta between consecutive data points in seconds */
    avg_timestamp_delta: f32,
    data_points: usize,
}

struct BlockInfo {
    block_size: usize,
    block_count: usize,
}

struct Index {
    timestamp_start: Vec<usize>,
    timestamp_end: Vec<usize>,
}

struct Series {
    name: String,
    schema: Schema,
    metadata: Metadata,
    block_info: BlockInfo,
    index: Index,
}

impl Series {
    fn push() {}
}

/******************************/

/// contains currently loaded blocks
struct BlockCache {
    loaded_blocks: Vec<Block>
}

/// contains all unprocessed operations
struct RedoLog {}

/// contains logic for loading block data from disk
struct BlockLoader {}

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
