use std::path::PathBuf;
use crate::engine::io::SyncPolicy;
use crate::engine::server::{Engine, SimpleServer, BlockLoader, Series};
use log::{info, debug, error};
use crate::server::protocol::{Command, Error, Response, Insert, Select, Between};
use serde::{Deserialize, Serialize};
use crate::engine::{Schema, Timestamp};
use crate::engine::index::TimestampIndex;
use std::time::Instant;
use std::fmt::Debug;
use crate::engine::Decoder;

pub mod protocol {
    use serde::{Serialize, Deserialize};

    #[derive(Serialize, Deserialize)]
    pub struct Between {
        pub min: Option<u64>,
        pub max: Option<u64>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Select {
        pub from: String,
        pub between: Between,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Insert<V> {
        pub to: String,
        pub value: V,
    }

    #[derive(Serialize, Deserialize)]
    pub enum Command<V> {
        Select(Select),
        Insert(Insert<V>),
        CreateSeries(String),
    }

    #[derive(Serialize, Deserialize)]
    pub enum Error {
        AuthError,
        InvalidQuery,
        TableNotFound,
        TableExists,
    }

    #[derive(Serialize, Deserialize)]
    pub enum Response<V> {
        Created,
        Inserted,
        Data(Vec<(u64, V)>),
    }
}

#[derive(Serialize, Deserialize)]
pub struct Settings {
    pub storage: PathBuf,
    pub block_cache_capacity: usize,
    pub block_sync_policy: SyncPolicy,
    pub index_sync_policy: SyncPolicy,

    /// Socket read timeout in milliseconds.
    pub socket_read_timeout: u64,

    /// Socket write timeout in milliseconds.
    pub socket_write_timeout: u64,

    /// Listen string in 'hostname:port' format.
    pub listen: String,
}

#[derive(Serialize, Deserialize)]
pub struct SeriesData<EncState> {
    id: usize,
    name: String,
    enc_state: EncState,
    blocks: usize,
    last_block_used_bytes: usize,
    last_timestamp: u64,
}

#[derive(Serialize, Deserialize)]
pub struct ServerData<EncState> {
    series: Vec<SeriesData<EncState>>,
    last_series_id: usize,
}

pub struct Server<S, V> where S: Schema<V> {
    settings: Settings,
    engine: Engine<S, V>,
}

impl<S, V, EncState> Server<S, V>
    where S: Schema<V, EncState=EncState>,
          V: Debug,
          for<'a> V: Copy + Serialize + Deserialize<'a>,
          for<'a> EncState: Serialize + Deserialize<'a> + Default + Copy
{
    pub fn new(settings: Settings) -> Self {
        let path = settings.storage.join("server.json");

        debug!("socket read timeout={}", settings.socket_read_timeout);
        debug!("socket write timeout={}", settings.socket_write_timeout);

        let server_data: ServerData<EncState> = if path.exists() {
            let reader = std::fs::File::open(path).unwrap();
            serde_json::from_reader(reader).unwrap()
        } else {
            ServerData {
                series: vec![],
                last_series_id: 0,
            }
        };

        let series = server_data.series.into_iter()
            .map(|s| {
                let index_file = settings.storage.join(format!("{}.idx", s.id));
                let index = TimestampIndex::create_or_load(&index_file, settings.index_sync_policy)
                    .unwrap();

                debug!("Loaded series: {} (ID {}) with {} blocks.", s.name, s.id, s.blocks);

                (s.name.to_owned(), Series {
                    id: s.id,
                    enc_state: s.enc_state,
                    timestamp_index: index,
                    blocks: s.blocks,
                    last_block_used_bytes: s.last_block_used_bytes,
                    last_timestamp: s.last_timestamp.into(),
                })
            })
            .collect();

        let blocks = BlockLoader::new(
            settings.storage.clone(),
            settings.block_cache_capacity,
            settings.block_sync_policy,
        );

        let mut engine = Engine {
            clock: Default::default(),
            storage: settings.storage.clone(),
            sync_policy: settings.block_sync_policy,
            last_series_id: server_data.last_series_id,
            series,
            blocks,
        };

        // verify all indices is correct if not, rebuild it.
        for (k, v) in engine.series.iter_mut() {
            if v.blocks != v.timestamp_index.len() {
                error!("Loaded index for series {} has incorrect length. Rebuilding...", k);
                let start = Instant::now();

                v.timestamp_index.invalidate();

                let mut dec_state = S::DecState::new(0.into());

                for block_id in 0..v.blocks {
                    let spec = v.create_block_spec(block_id);
                    let block = engine.blocks.acquire_block(spec);

                    let mut read_bytes = 0;
                    let written_bytes = if block_id == v.blocks - 1 { v.last_block_used_bytes } else { block.data_len() };

                    let mut min: Timestamp = std::u64::MAX.into();
                    let mut max: Timestamp = std::u64::MIN.into();

                    while read_bytes < written_bytes {
                        let offset_buff = &block.data[read_bytes..];
                        let (point, rb) = S::decode(&mut dec_state, offset_buff);

                        min = min.min(*point.timestamp());
                        max = max.max(*point.timestamp());

                        read_bytes += rb
                    }

                    v.timestamp_index.set_max(block_id, max);
                    v.timestamp_index.set_min(block_id, min);
                }

                v.timestamp_index.write_dirty_part();
                info!("Rebuilt index in {} secs...", start.elapsed().as_secs_f32())
            }
        }

        Server {
            engine,
            settings
        }
    }

    /// Persists metadata (information about series) to the disk.
    fn persist_metadata(&self) {
        let path = self.settings.storage.join("server.json");
        let data = ServerData {
            series: self.engine.series.iter().map(|(name, series)| {
                SeriesData {
                    id: series.id,
                    name: name.clone(),
                    enc_state: series.enc_state,
                    blocks: series.blocks,
                    last_block_used_bytes: series.last_block_used_bytes,
                    last_timestamp: (&series.last_timestamp).into(),
                }
            }).collect(),
            last_series_id: self.engine.last_series_id,
        };

        if let Err(e) = std::fs::write(path, serde_json::to_string(&data).unwrap()) {
            error!("Cannot save server metadata! {}", e)
        }
    }

    /// Handles the command on the server and returns the appropriate
    /// result. The calling method will handle the result (usually
    /// writes the result into a socket).
    pub fn handle_command(&mut self, cmd: Command<V>) -> Result<Response<V>, Error> {
        match cmd {
            Command::Select(Select { from, between }) => {
                let Between { min, max } = between;
                let data = self.engine
                    .retrieve_points(&from,
                                     min.map(|x| x.into()),
                                     max.map(|x| x.into()),
                    )
                    .map_err(|_| Error::TableNotFound)?
                    .iter()
                    .map(|p| (p.timestamp().into(), *p.value()))
                    .collect();
                self.persist_metadata();
                Ok(Response::Data(data))
            }
            Command::Insert(Insert { to, value }) => {
                self.engine
                    .insert_point(&to, value)
                    .map_err(|_| Error::TableNotFound)?;
                self.persist_metadata();
                Ok(Response::Inserted)
            }
            Command::CreateSeries(name) => {
                self.engine
                    .create_series(&name)
                    .map_err(|_| Error::TableExists)?;
                self.persist_metadata();
                Ok(Response::Created)
            }
        }
    }
}

#[cfg(test)]
mod tests {}