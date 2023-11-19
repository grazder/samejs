importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js");

const ONNX_MODEL_PATH = '/model.onnx'
const ONNX_WASM_PATH = '/wasm_files/ort-wasm.wasm'

class NoiseCancellationWorker {
  model;

  constructor() {
    this.init();
  }

  async init() {
    await this.initModel();
    await this.registerListeners();
  }

  async initModel() {
    ort.env.debug = false;
    ort.env.wasm.simd = false;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
        'ort-wasm.wasm': self.location.origin + ONNX_WASM_PATH,
    };    

    this.model = await ort.InferenceSession.create(self.location.origin + ONNX_MODEL_PATH, {
      executionProviders: ['wasm'],
    });
  }

  async registerListeners() {
    const { port1: portToWorklet, port2: portToWorker } = new MessageChannel();

    portToWorklet.onmessage = async (e) => {
      const { input } = e.data;

      const inputArray = new Float32Array(input);
      const { buffer } = await this.process(inputArray);

      portToWorklet.postMessage(
        {
          output: buffer,
        },
        [buffer],
      );
    };
    self.postMessage(portToWorker, [portToWorker]);
  }

  async process(input_array) {
    const inputFrame = new ort.Tensor('float32', input_array, [input_array.length]);

    const outputMap = await this.model.run({
      x: inputFrame,
    });
    
    return outputMap.out.data;
  }
}

new NoiseCancellationWorker();