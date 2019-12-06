use std::path::PathBuf;
use crate::engine::io::SyncPolicy;
use crate::engine::server::{Engine, SimpleServer, BlockLoader, Series};
use std::net::{TcpListener, TcpStream, SocketAddr, Shutdown};
use log::{info, debug};
use crate::server::protocol::{Command, Error, Response, Insert, Select, Between};
use serde::{Deserialize, Serialize};
use crate::engine::{Schema};
use std::io::Write;
use crate::engine::index::TimestampIndex;

mod protocol {
    use serde::{Serialize, Deserialize};

    #[derive(Serialize, Deserialize)]
    pub struct Between {
        min: Option<u64>,
        max: Option<u64>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Select<'a> {
        from: &'a str,
        between: Between,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Insert<'a, V> {
        to: &'a str,
        value: V,
    }

    #[derive(Serialize, Deserialize)]
    pub enum Command<'a, V> {
        Select(Select<'a>),
        Insert(Insert<'a, V>),
        CreateSeries(&'a str),
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
    storage: PathBuf,
    block_cache_capacity: usize,
    query_cache_capacity: usize,
    block_sync_policy: SyncPolicy,
    block_write_policy: SyncPolicy,
    index_sync_policy: SyncPolicy,
    index_write_policy: SyncPolicy,
    listen: String,
}

#[derive(Serialize, Deserialize)]
pub struct SeriesData<S, V, T> where S: Schema<V, EncState=T>, T: Default {
    id: usize,
    name: String,
    enc_state: S::EncState,
    blocks: usize,
    last_block_used_bytes: usize,
    last_timestamp: u64,
}

#[derive(Serialize, Deserialize)]
pub struct ServerData<S, V, T> where S: Schema<V, EncState=T>, T: Default {
    series: Vec<SeriesData<S, V, T>>,
    last_series_id: usize,
}

pub struct Server<S, V> where S: Schema<V> {
    settings: Settings,
    engine: Engine<S, V>,
    tcp: TcpListener,
}

impl<'a, S, V, T> Server<S, V>
    where S: Schema<V, EncState=T> + Deserialize<'a>,
          V: Serialize + Deserialize<'a> + Copy,
          T: Serialize + Deserialize<'a> + Default
{
    pub fn new(settings: Settings) -> Self {
        let file = std::fs::read_to_string(settings.storage.join("server.json")).unwrap();
        let series_data: ServerData<S, V, T> = serde_json::from_str(&file).unwrap();

        let series = series_data.series.into_iter()
            .map(|s| {
                let index_file = settings.storage.join(format!("{}.idx", s.id));
                let index = TimestampIndex ::create_or_load(&index_file, settings.index_sync_policy)
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
            storage: Default::default(),
            sync_policy: SyncPolicy::Immediate,
            last_series_id: 0,
            series,
            blocks,
        };

        let tcp = TcpListener::bind(settings.listen)
            .unwrap();

        Server {
            engine,
            settings,
            tcp,
        }
    }

    pub fn listen(&mut self) -> Self {
        info!("TCP Server listening on {:?}...", self.tcp.local_addr().unwrap());
        loop {
            let (mut stream, remote) = self.tcp.accept().unwrap();
            debug!("New client from {:?}", remote);
            match self.handle_client(&mut stream, &remote) {
                Ok(response) => {
                    let response = serde_json::to_string(&response).unwrap();
                    stream.write_all(response.as_bytes());
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

    fn handle_client(&mut self, stream: &'a mut std::net::TcpStream, remote: &SocketAddr) -> Result<Response<V>, Error> {
        // first we need to authenaticate the client
        self.authenticate(stream)?;

        // then we read command from stream and fulfill it returing
        // the result of the command's execution
        match self.read_command(stream)? {
            Command::Select(Select { from, between }) => {
                let Between { min, max } = between;
                let data = self.engine
                    .retrieve_points(from,
                                     min.map(|x| x.into()),
                                     max.map(|x| x.into()),
                    )
                    .map_err(|e| Error::TableNotFound)?
                    .iter()
                    .map(|p| (p.timestamp().into(), *p.value()))
                    .collect();
                Ok(Response::Data(data))
            }
            Command::Insert(Insert { to, value }) => {
                self.engine
                    .insert_point(to, value)
                    .map_err(|_| Error::TableNotFound)?;
                Ok(Response::Inserted)
            }
            Command::CreateSeries(name) => {
                self.engine
                    .create_series(name)
                    .map_err(|_| Error::TableNotFound)?;
                Ok(Response::Created)
            }
        }
    }

    fn read_command<'b>(&mut self, stream: &'b mut TcpStream) -> Result<Command<'b, V>, Error> {
        let mut de = serde_json::Deserializer::from_reader(stream);
        Command::deserialize(&mut de).map_err(|x| Error::InvalidQuery)
    }

    fn authenticate(&mut self, _stream: &mut TcpStream) -> Result<(), Error> {
        Ok(())
    }
}