use std::path::PathBuf;
use tsdb::engine::f32::F32;
use tsdb::engine::io::SyncPolicy;
use tsdb::server::{Settings, Server};
use tsdb::server::protocol::{Command, Insert, Select, Between};


fn main() {
    simple_logger::init_with_level(log::Level::Debug).unwrap();

    println!("{}", serde_json::to_string(&Command::<f32>::CreateSeries("default".into())).unwrap());
    println!("{}", serde_json::to_string(&Command::Insert(Insert {
        to: "default".into(),
        value: 3.14,
    })).unwrap());
    println!("{}", serde_json::to_string(&Command::<f32>::Select(Select {
        from: "default".into(),
        between: Between { min: None, max: None },
    })).unwrap());

    let settings = Settings {
        storage: PathBuf::from("./storage/"),
        block_cache_capacity: 1024,
        block_sync_policy: SyncPolicy::Never,
        index_sync_policy: SyncPolicy::Never,
        listen: "0.0.0.0:9087".to_string(),
    };

    let mut server = Server::<F32, f32>::new(settings);
    server.listen();
}