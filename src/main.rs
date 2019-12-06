use std::path::PathBuf;
use std::time::Instant;
use tsdb::engine::server::{Engine, SimpleServer};
use tsdb::engine::f32::F32;
use tsdb::engine::io::SyncPolicy;


fn main() {
    simple_logger::init_with_level(log::Level::Debug).unwrap();

    let mut s: Engine<F32, f32> = Engine::new(
        PathBuf::from("./storage/"),
        1024,
        SyncPolicy::Never,
    );

    s.create_series("default");
    s.insert_point("default", 3.14);

    measure("insert 50000 records", || {
        for f in 0..50000 {
            s.insert_point("default", f as f32);
        }
    });

    let pts = s.retrieve_points("default", None, None).ok().unwrap();
    println!("{:#?}", &pts[0..1]);
}

#[inline]
fn measure<F: FnOnce() -> ()>(label: &str, f: F) {
    let start = Instant::now();
    f();
    println!("{} took {}s", label, start.elapsed().as_secs_f32())
}