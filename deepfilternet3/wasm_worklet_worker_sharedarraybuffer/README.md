For successful wasm convertion you need:
1. Install Rust
2. `cargo install wasm-pack`

(You can find it in workflows also - https://github.com/Rikorose/DeepFilterNet/blob/main/.github/workflows/build_wasm.yml)

To get WASM package and model:
```
git clone https://github.com/Rikorose/DeepFilterNet/
cd DeepFilterNet
bash scripts/build_wasm_package.sh
cp -r libdf/pkg ../samejs/deepfilternet3/wasm_worklet_worker_sharedarraybuffer/
cp models/DeepFilterNet3_onnx.tar.gz ../samejs/deepfilternet3/wasm_worklet_worker_sharedarraybuffer/
cd ../samejs/deepfilternet3/wasm_worklet_worker_sharedarraybuffer/
```