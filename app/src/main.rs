use std::env;
use std::io::Write;
use std::net::Ipv4Addr;
use std::process::{Command, Stdio};
use warp::Filter;

use std::io::{self, Read};
use wasmedge_tensorflow_interface;

pub fn image_process(buf: &Vec<u8>) -> String {
    let model_data: &[u8] =
        include_bytes!("models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_quant.tflite");
    let labels = include_str!("models/mobilenet_v1_1.0_224/labels_mobilenet_quant_v1_224.txt");

    let flat_img = wasmedge_tensorflow_interface::load_jpg_image_to_rgb8(&buf, 224, 224);

    let mut session = wasmedge_tensorflow_interface::Session::new(
        &model_data,
        wasmedge_tensorflow_interface::ModelType::TensorFlowLite,
    );
    session
        .add_input("input", &flat_img, &[1, 224, 224, 3])
        .run();
    let res_vec: Vec<u8> = session.get_output("MobilenetV1/Predictions/Reshape_1");

    let mut i = 0;
    let mut max_index: i32 = -1;
    let mut max_value: u8 = 0;
    while i < res_vec.len() {
        let cur = res_vec[i];
        if cur > max_value {
            max_value = cur;
            max_index = i as i32;
        }
        i += 1;
    }
    // println!("{} : {}", max_index, max_value as f32 / 255.0);

    let mut confidence = "could be";
    if max_value > 200 {
        confidence = "is very likely";
    } else if max_value > 125 {
        confidence = "is likely";
    } else if max_value > 50 {
        confidence = "could be";
    }

    let mut label_lines = labels.lines();
    for _i in 0..max_index {
        label_lines.next();
    }

    let class_name = label_lines.next().unwrap().to_string();
    if max_value > 50 {
        format!(
            "It {} a <a href='https://www.google.com/search?q={}'>{}</a> in the picture",
            confidence.to_string(),
            class_name,
            class_name
        )
    } else {
        format!("It does not appears to be any food item in the picture.")
    }
}

/*
pub fn image_process(buf: &Vec<u8>) -> String {
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
    ans.to_string()
}
*/

#[tokio::main]
pub async fn run_server(port: u16) {
    pretty_env_logger::init();

    let home = warp::get()
        .and(warp::path::end())
        .and(warp::fs::file("./static/home.html"));

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
