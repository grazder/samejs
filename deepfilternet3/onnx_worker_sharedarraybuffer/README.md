# ONNX Example

To get `denoiser_model.ort` you should go here - https://github.com/grazder/DeepFilterNet/tree/torchDF-changes/torchDF. Do all the installing steps and then:

```
poetry run python model_onnx_export.py --test --performance --inference-path examples/A1CIM28ZUCA8RX_M_Street_Near_Regular_SP_Mobile_Primary.wav --ort
cp denoiser_model.ort /path/to/samejs/deepfilternet3/onnx_worklet_worker_sharedarraybuffer/
```

To get wasm_files:
```
cd /path/to/samejs/deepfilternet3/onnx_worklet_worker_sharedarraybuffer/
mkdir wasm_files
wget -O wasm_files/ort-wasm.wasm https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort-wasm.wasm
wget -O wasm_files/ort-wasm-simd.wasm https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort-wasm-simd.wasm
```

To run demo:
```
node node_server.js
```