class AudioWorkletExample extends AudioWorkletProcessor {
    portToWorker;

    denoiseFrameSize = 1920;
    workletFrameSize = 128;
    ringBufferLength = 10000;
    
    constructor(options) {
      super();

      this.inputBuffer = new RingBuffer(this.ringBufferLength, 1);
      this.outputBuffer = new RingBuffer(this.ringBufferLength, 1);
      this.planarBuffer = new Float32Array(this.denoiseFrameSize);

      this.port.onmessage = (e) => {
          if (e.data instanceof MessagePort) {
              this.portToWorker = e.data;
              this.portToWorker.onmessage = (event) => {
                  const { output } = event.data;
                  const outputArray = new Float32Array(output);
                  this.outputBuffer.push([outputArray]);
              }
          };
      };
    }

    process(inputs, outputs) {
        if (!this.portToWorker) {
          return true;
        }

        this.inputBuffer.push(inputs[0]);

        if (this.inputBuffer.framesAvailable > this.denoiseFrameSize) {
            this.inputBuffer.pull([this.planarBuffer]);

            this.portToWorker.postMessage([this.planarBuffer]);
        }
        
        if (this.outputBuffer.framesAvailable >= this.workletFrameSize) {
            this.outputBuffer.pull(outputs[0]);
        } else {
            console.warn("WARNING! AudioWorklet: Not enogh samples");
        }

        return true;
    }  
}

registerProcessor("audio-worklet-example", AudioWorkletExample);
