use log::{debug, error, info, warn};
use std::io;
use std::path::PathBuf;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::AsyncRead;
use tsdb::engine::f32::F32;
use tsdb::engine::io::SyncPolicy;
use tsdb::server::{Server, Settings};
use futures::future::poll_fn;

#[tokio::main]
async fn main() -> io::Result<()> {
    simple_logger::init_with_level(log::Level::Debug).unwrap();

    let settings = Settings {
        storage: PathBuf::from("./storage/"),
        block_cache_capacity: 1024 * 8, // 32 mb
        block_sync_policy: SyncPolicy::Never,
        index_sync_policy: SyncPolicy::Never,
        socket_read_timeout: 2000,
        socket_write_timeout: 5000,
        listen: "0.0.0.0:9087".to_string(),
    };
    let mut listener = TcpListener::bind(&settings.listen).await?;
    info!("Listening on {}", &settings.listen);
    let db = Server::<F32, f32>::new(settings);

    loop {
        let (socket, addr) = listener.accept().await?;
        info!("New client from {:?}", addr);

        handle_client(socket);
    }
}

async fn handle_client(socket: TcpStream) {
        let mut buf = [0; 10];
        let result = poll_fn(|cx| {
            socket.poll_peek(cx, &mut buf)
        }).await;

    let mut de = serde_json::from_str(result.unwrap());
}
