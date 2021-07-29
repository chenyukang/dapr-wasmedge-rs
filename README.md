# dapr-wasmedge

## Directories

`server` : Use [Wrap](https://github.com/seanmonstar/warp) to implement a simple Web server, response for static file request and image classify API.

`image-classification`: Will build a `wasm32-wasi` target file to classfy image with [WasmEdge](https://github.com/WasmEdge/WasmEdge/) and Tensorflow.

`lib`: Some library and binary files for `image-classification`.

`config`: Sample configuration files for [Dapr](https://dapr.io/)

`static`: Static files(HTML, JavaScript, and CSS files for website)

## Installation

1. [Install Dapr](https://docs.dapr.io/getting-started/install-dapr-cli/)
2. Compile it with `./compile.sh`

## Run 

```shell
sudo dockerd  
sudo dapr run --app-id image-classify --app-port 8000 ./app/target/debug/app
```

