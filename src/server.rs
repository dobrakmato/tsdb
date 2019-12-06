use std::path::PathBuf;
use crate::engine::io::SyncPolicy;
use crate::engine::server::{Engine, SimpleServer, BlockLoader, Series};
use std::net::{TcpListener, TcpStream, SocketAddr, Shutdown};
use log::{info, debug, warn, error};
use crate::server::protocol::{Command, Error, Response, Insert, Select, Between};
use serde::{Deserialize, Serialize};
use crate::engine::Schema;
use std::io::Write;
use crate::engine::index::TimestampIndex;
use std::time::Duration;

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
    tcp: TcpListener,
}

impl<S, V, EncState> Server<S, V>
    where S: Schema<V, EncState=EncState>,
          for<'a> V: Copy + Serialize + Deserialize<'a>,
          for<'a> EncState: Serialize + Deserialize<'a> + Default + Copy
{
    pub fn new(settings: Settings) -> Self {
        let path = settings.storage.join("server.json");

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

        let engine = Engine {
            clock: Default::default(),
            storage: settings.storage.clone(),
            sync_policy: settings.block_sync_policy,
            last_series_id: server_data.last_series_id,
            series,
            blocks,
        };

        let tcp = TcpListener::bind(settings.listen.clone())
            .unwrap();

        Server {
            engine,
            settings,
            tcp,
        }
    }

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

    pub fn listen(&mut self) -> Self {
        info!("TCP Server listening on {:?}...", self.tcp.local_addr().unwrap());
        loop {
            let (mut stream, remote) = self.tcp.accept().unwrap();
            debug!("New client from {:?}", remote);
            stream.set_read_timeout(Some(Duration::from_millis(self.settings.socket_read_timeout)))
                .unwrap();
            stream.set_write_timeout(Some(Duration::from_millis(self.settings.socket_write_timeout)))
                .unwrap();
            match self.handle_client(&mut stream, &remote) {
                Ok(response) => {
                    let response = serde_json::to_string(&response).unwrap();
                    stream.write_all(response.as_bytes()).unwrap();
                    stream.shutdown(Shutdown::Both).unwrap();
                }
                Err(err) => {
                    let err = serde_json::to_string(&err).unwrap();
                    stream.write_all(err.as_bytes()).unwrap();
                    stream.shutdown(Shutdown::Both).unwrap()
                }
            }
        }
    }

    fn handle_client(&mut self, stream: &mut std::net::TcpStream, _remote: &SocketAddr) -> Result<Response<V>, Error> {
        // first we need to authenticate the client
        self.authenticate(stream)?;

        let mut de = serde_json::Deserializer::from_reader(stream);
        let cmd = Command::deserialize(&mut de)
            .map_err(|e| {
                warn!("Invalid query submitted: {}", e);
                Error::InvalidQuery
            })?;

        // then we read command from stream and fulfill it returning
        // the result of the command's execution
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

    fn authenticate(&mut self, _stream: &mut TcpStream) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {}