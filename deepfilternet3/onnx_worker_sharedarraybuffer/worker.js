// importScripts add package into exports, so it goes first
const exports = {};

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js");
ort = exports.ort

// onnxruntime
ort.env.debug = false;
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = 1;

// set global logging level
ort.env.logLevel = 'verbose';

let base_url;

// Read some float32 pcm from the queue, convert to int16 pcm, and push it to
// our global queue.
async function readFromQueue() {
  if (this._audio_reader.available_read() < 480) {
    return 0;
  }
  
  const samples_read = this._audio_reader.dequeue(this.rawStorage);
  const input_frame = new ort.Tensor("float32", 
    this.rawStorage.subarray(0, samples_read), [samples_read]
  )

  start = performance.now();
  const outputMap = await self._model.run({
    input_frame: input_frame,
    states: this.states,
    atten_lim_db: this.atten_lim_db
  });
  end = performance.now() - start;
  // console.log(end);

  this.states = outputMap.out_states;

  if (this._audio_writer.enqueue(outputMap.enhanced_audio_frame.data) !== 480) {
      console.log("worker underrun: the audioworklet doesn't dequeue fast enough!");
  }

  return samples_read;
}

function linspace(startValue, stopValue, cardinality) {
  var arr = [];
  var step = (stopValue - startValue) / (cardinality - 1);
  for (var i = 0; i < cardinality; i++) {
    arr.push(startValue + (step * i));
  }
  return arr;
}

onmessage = async (e) => {
  switch (e.data.command) {
    case "init": {
      this._audio_reader = new exports.AudioReader(
        new RingBuffer(e.data.rawSab, Float32Array)
      );

      this._audio_writer = new exports.AudioWriter(
        new RingBuffer(e.data.denoisedSab, Float32Array)
      )

      // The sample-rate of the audio stream read from the queue.
      this.sampleRate = e.data.sampleRate;
      base_url = e.data.base_url
      hop_size = e.data.hop_size
      state_size = e.data.state_size

      // override path of wasm files - for each file
      ort.env.wasm.wasmPaths = {
        'ort-wasm.wasm': base_url + 'wasm_files/ort-wasm.wasm',
        'ort-wasm-simd.wasm':base_url + 'wasm_files/ort-wasm-simd.wasm',
        'ort-wasm-threaded.wasm': base_url + 'wasm_files/ort-wasm-threaded.wasm',
        'ort-wasm-simd-threaded.wasm': base_url + 'wasm_files/ort-wasm-simd-threaded.wasm',
      };
      
      // A smaller staging array to copy the audio samples from, before conversion
      // to uint16. It's size is 4 times less than the 1 second worth of data
      // that the ring buffer can hold, so it's 250ms, allowing to not make
      // deadlines:
      // staging buffer size = ring buffer size / sizeof(float32) / stereo / 4
      this.rawStorage = new Float32Array(hop_size);

      // Initializing ORT model
      this._model = await ort.InferenceSession.create(base_url + e.data.onnx_path, { executionProviders: ['wasm'], });
      console.log(`ONNX model loaded...`)

      // States
      this.states = new ort.Tensor("float32", new Array(state_size).fill(0), [state_size]);
      this.atten_lim_db = new ort.Tensor("float32", new Array(1).fill(0), [1]);

      // States
      this.states = new ort.Tensor("float32", new Array(state_size).fill(0), [state_size]);
      this.atten_lim_db = new ort.Tensor("float32", new Array(1).fill(0), [1]);

      // init run for onnx
      const input_frame = new ort.Tensor("float32", 
        this.rawStorage.subarray(0, hop_size), [hop_size]
      )
      const outputMap = await self._model.run({
        input_frame: input_frame,
        states: this.states,
        atten_lim_db: this.atten_lim_db
      });

      postMessage(0);

      // interval = setInterval(readFromQueue, 0);

      while (true){
        await readFromQueue();
      }
      break;
    }
    case "stop": {
      clearInterval(interval);
      break;
    }
    default: {
      console.log(e.data)
      throw Error("Case not handled");
    }
  }
};