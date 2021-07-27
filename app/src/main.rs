use bytes::Bytes;
use std::env;
use std::io::Write;
use std::net::Ipv4Addr;
use std::process::{Command, Stdio};
use warp::{Filter, Rejection};

pub fn image_process(buf: &Vec<u8>) -> String {
    println!("buf: {:?}", buf.len());
    let mut child = Command::new("./lib/wasmedge-tensorflow-lite")
        .arg("./lib/classify.so")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to execute child");
    {
        // limited borrow of stdin
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin.write_all(buf).expect("failed to write to stdin");
    }
    let output = child.wait_with_output().expect("failed to wait on child");
    let ans = String::from_utf8_lossy(&output.stdout);
    println!("ans: {:?}", ans);
    ans.to_string()
}

#[tokio::main]
pub async fn run_server(port: u16) {
    pretty_env_logger::init();

    let home = warp::get()
        .and(warp::path::end())
        .and(warp::fs::file("./home.html"));

    // dir already requires GET...
    let statics = warp::get()
        .and(warp::path("static"))
        .and(warp::fs::dir("./static/"));

    let image = warp::post()
        .and(warp::path("api"))
        .and(warp::path("hello"))
        .and(warp::body::bytes())
        .map(|bytes: bytes::Bytes| {
            //println!("bytes = {:?}", bytes);
            let v: Vec<u8> = bytes.iter().map(|&x| x).collect();
            println!("v: len {}", v.len());
            let res = image_process(&v);
            println!("result: {}", res);
            Ok(Box::new(res))
        });

    // GET / => home.html
    // GET /static/... => ./static/..
    let routes = home.or(statics).or(image);
    let routes = routes
        .recover(handle_rejection)
        .with(warp::cors().allow_any_origin());

    let log = warp::log("dapr_wasm");
    let routes = routes.with(log);
    println!("listen to : {} ...", port);
    warp::serve(routes).run((Ipv4Addr::UNSPECIFIED, port)).await
}

fn log_body() -> impl Filter<Extract = (), Error = Rejection> + Copy {
    warp::body::bytes()
        .map(|b: Bytes| {
            let v: Vec<u8> = b.iter().map(|&x| x).collect();
            println!("result: {}", v.len());
        })
        .untuple_one()
}

async fn handle_rejection(
    err: warp::Rejection,
) -> Result<impl warp::Reply, std::convert::Infallible> {
    Ok(warp::reply::json(&format!("{:?}", err)))
}

fn main() {
    let port_key = "FUNCTIONS_CUSTOMHANDLER_PORT";
    let _port: u16 = match env::var(port_key) {
        Ok(val) => val.parse().expect("Custom Handler port is not a number!"),
        Err(_) => 8000,
    };

    run_server(_port);
}
