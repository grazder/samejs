let gainNode;
let gainRange;
let audioContext = null;
let sampleRate = 48000;
let worker;
let onnx_path = './denoiser_model.ort'

window.addEventListener("load", (event) => {
  document.getElementById("toggle").addEventListener("click", toggleSound);

  gainRange = document.getElementById("gain");  
  gainRange.oninput = () => {
    gainNode.gain.value = gainRange.value;
  };
  
  gainRange.disabled = true;
});

async function toggleSound(event) {
  if (!audioContext) {
    audioDemoStart();

    gainRange.disabled = false;
  } else {
    gainRange.disabled = true;
    worker.postMessage({"command": "stop"})

    await audioContext.close();
    audioContext = null;
  }
}

async function setupWorker(rawSab, denoisedSab) {
  // The Web Worker can receive two commands: 
  // - on "init", it starts periodically reading from the queue and
  //  accumulating audio data.
  // - on "stop", it takes all this accumulated audio data, converts to PCM16
  // instead of float32 and turns the stream into a WAV file, sending it back
  // to the main thread to offer it as download.

  URLFromFiles(['worker.js', 'ringbuffer.js']).then((e) => {
      worker = new Worker(e);

      worker.onmessage = (e) => {
        const { type } = e.data;

        switch (type) {
          case "FETCH_WASM": {
            fetch("/pkg/df_bg.wasm")
              .then((response) => response.arrayBuffer())
              .then((bytes) => {
                fetch('/DeepFilterNet3_onnx.tar.gz').then((response) => response.arrayBuffer())
                .then((model_bytes) => {
                  worker.postMessage({
                    command: "init",
                    bytes: bytes,
                    base_url: document.baseURI,
                    model_bytes: model_bytes,
                    rawSab: rawSab,
                    denoisedSab: denoisedSab
                  });
                });
              });
              console.log("fetching...")
            break;
          }
          case "SETUP_AWP": {
            setupWebAudio(rawSab, denoisedSab);
          }
          default: {
            break;
          }
        }
      }
      
      // worker.postMessage({
      //   command: "init", 
        // rawSab: rawSab,
        // denoisedSab: denoisedSab,
      //   sampleRate: sampleRate,
      //   base_url: document.baseURI,
      //   onnx_path: onnx_path,
      //   hop_size: 480,
      //   state_size: 45304
      // });
  });
};

async function setupWebAudio(rawSab, denoisedSab) {
  audioContext.resume();

  gainNode = audioContext.createGain()

  URLFromFiles(['audio-processor.js', 'ringbuffer.js']).then((e) => {
      audioContext.audioWorklet.addModule(e)
        .then(() => getLiveAudio(audioContext))
        .then((liveIn) => {
            // After the resolution of module loading, an AudioWorkletNode can be constructed.
            let audioProcesser = new AudioWorkletNode(audioContext, 'random-audio-processor', 
                {
                  processorOptions: {
                    rawSab: rawSab,
                    denoisedSab: denoisedSab
                  }
                }
            )
            // AudioWorkletNode can be interoperable with other native AudioNodes.
            liveIn.connect(audioProcesser).connect(gainNode).connect(audioContext.destination)
        })
        .catch(e => console.error(e))
  });
}

async function audioDemoStart() {
  audioContext = new AudioContext({sampleRate: sampleRate})

  var rawSab = RingBuffer.getStorageForCapacity(sampleRate * 2, Float32Array);
  var denoisedSab = RingBuffer.getStorageForCapacity(sampleRate * 2, Float32Array);

  setupWorker(rawSab, denoisedSab);
}

function getLiveAudio(audioContext) {
  return navigator.mediaDevices.getUserMedia({
          audio: true
      })
      .then(stream => audioContext.createMediaStreamSource(stream))
}