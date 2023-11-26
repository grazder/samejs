// importScripts add package into exports, so it goes first
const exports = {};

let frame_length;
let df_model;

// Read some float32 pcm from the queue, convert to int16 pcm, and push it to
// our global queue.
async function readFromQueue() {
  if (this._audio_reader.available_read() < frame_length) {
    return 0;
  }

  const samples_read = this._audio_reader.dequeue(this.rawStorage);
  let input_frame = this.rawStorage.subarray(0, samples_read)
  
  let output_frame = wasm_bindgen.df_process_frame(df_model, input_frame);

  if (this._audio_writer.enqueue(output_frame) !== frame_length) {
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
      // A smaller staging array to copy the audio samples from, before conversion
      // to uint16. It's size is 4 times less than the 1 second worth of data
      // that the ring buffer can hold, so it's 250ms, allowing to not make
      // deadlines:
      // staging buffer size = ring buffer size / sizeof(float32) / stereo / 4

      base_url = e.data.base_url
      importScripts(base_url + '/pkg/df.js');
      wasm_bindgen.initSync(e.data.bytes);

      const uint8Array = new Uint8Array(e.data.model_bytes);
      df_model = wasm_bindgen.df_create(uint8Array, 100.0);
      console.log('df_model loaded...');

      frame_length = wasm_bindgen.df_get_frame_length(df_model)
      this.rawStorage = new Float32Array(frame_length);

      interval = setInterval(readFromQueue, 0);
      postMessage({ type: "SETUP_AWP" })

      // while (true){
      //   await readFromQueue();
      // }
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

postMessage({ type: "FETCH_WASM" });