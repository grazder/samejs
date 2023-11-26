// In `script.js` we connect our AudioWorklet and Worker.

// Here we create the worker so that it is visible anywhere in the code.
let worker;
let audioContext = null;

// Let's create a "start" button listener to turn your pipeline on.
window.addEventListener("load", (event) => {
  document.getElementById("start").addEventListener("click", startPipeline);
});

window.addEventListener("load", (event) => {
  document.getElementById("stop").addEventListener("click", stopPipeline);
});

async function stopPipeline(enent) {
    await audioContext.close();
    audioContext = null;
}

// Let's start our pipeline
async function startPipeline(event) {
    // Let's create AudioContex first.
    // Sample Rate is an important parameter because most ML models are trained on data with a single sample rate.
    audioContext = new AudioContext({sampleRate: 48000});

    const workletUrl = await URLFromFiles([
        'audio-worklet.js', 'data_structures/ringbuffer.js'
    ])

    // 1. Creating AudioWorklet from `audio_worklet.js`
    await audioContext.audioWorklet.addModule(workletUrl);

    // 2. Getting input from microphone
    stream = await navigator.mediaDevices.getUserMedia({
        audio: true
    });

    // 3. Create a microphone input stream and add it as an audioContext source.
    liveIn = audioContext.createMediaStreamSource(stream);

    // 4. Simple creating a worker in which our pipeline will run.
    worker = new Worker('worker.js');
    
    // 5. Creating AudioWorkletNode 
    audioProcessor = new AudioWorkletNode(audioContext, 'audio-worklet-example')

    // 6. Forward message port to 
    worker.onmessage = (e) => {
        audioProcessor.port.postMessage(e.data, {transfer: [e.data]});
    }

    // 7. Connecting all the nodes 
    // Microphone Stream (liveIn) -> AudioWorklet processor (audioProcessor) -> Headphones (default destination)
    liveIn.connect(audioProcessor).connect(audioContext.destination)
}

function URLFromFiles(files) {
    const promises = files.map((file) =>
      fetch(file).then((response) => response.text())
    );
  
    return Promise.all(promises).then((texts) => {
      const text = texts.join("");
      const blob = new Blob([text], { type: "application/javascript" });
  
      return URL.createObjectURL(blob);
    });
  }