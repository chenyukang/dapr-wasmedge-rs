cd image-classification
cargo build --target wasm32-wasi
cp ./target/wasm32-wasi/debug/classify.wasm ../lib/
cd ../

