// In `script.js` we connect our AudioWorklet and Worker.
// Let's create AudioContex first.
// Sample Rate is an important parameter because most ML models are trained on data with a single sample rate.
const audioContext = new AudioContext({sampleRate: 48000});

// Here we create the worker so that it is visible anywhere in the code.
let worker;

// Let's create a "toggle" button listener to turn your pipeline on.
window.addEventListener("load", (event) => {
  document.getElementById("start").addEventListener("click", toggleSound);
});

// Let's start our pipeline
async function startPipeline(event) {
    // 1. Creating AudioWorklet from `audio_worklet.js`
    await audioContext.audioWorklet.addModule('audio_worklet.js')

    // 2. Getting input from microphone
    stream = navigator.mediaDevices.getUserMedia({
        audio: true
    })

    // 3. Create a microphone input stream and add it as an audioContext source.
    liveIn = audioContext.createMediaStreamSource(stream)

    // 4. Simple creating a worker in which our pipeline will run.
    worker = new Worker('worker.js');
    
    // 5. Creating AudioWorkletNode 
    let audioProcessor = new AudioWorkletNode(audioContext, 'audio-worklet-example')

    // 6. Forward message port to 
    worker.onmessage = (e) => {
        audioProcessor.port.postMessage(e.data, {transfer: [e.data]});
    }

    // 7. Connecting all the nodes 
    // Microphone Stream (liveIn) -> AudioWorklet processor (audioProcessor) -> Headphones (default destination)
    liveIn.connect(audioProcessor).connect(audioContext.destination)
}