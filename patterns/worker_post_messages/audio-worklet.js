class AudioWorkletExample extends AudioWorkletProcessor {
    portToWorker;

    denoiseFrameSize = 1920;
    workletFrameSize = 128;
    ringBufferLength = 10000;
    channel_count = 1;
    
    constructor(options) {
      super();

      this.inputBuffer = new RingBuffer(this.ringBufferLength, this.channel_count);
      this.outputBuffer = new RingBuffer(this.ringBufferLength, this.channel_count);
      this.planarBuffer = new Float32Array(this.denoiseFrameSize);

      this.port.onmessage = (e) => {
          if (e.data instanceof MessagePort) {
              this.portToWorker = e.data;
              this.portToWorker.onmessage = (event) => {
                  const { output } = event.data;
                  const outputArray = new Float32Array(output);
                  this.outputBuffer.push([outputArray]);
              }
          }
          else {
                this.portToWorker = null;
          };
      };
    }

    process(inputs, outputs) {
        const input = inputs[0];
        const output = outputs[0];

        if (!this.portToWorker) {
            for (let channel = 0; channel < this.channel_count; ++channel) {
              output[channel].set(input[channel]);
            }
        
            return true;
        }

        this.inputBuffer.push(input);

        if (this.inputBuffer.framesAvailable > this.denoiseFrameSize) {
            this.inputBuffer.pull([this.planarBuffer]);

            this.portToWorker.postMessage([this.planarBuffer]);
        }
        
        if (this.outputBuffer.framesAvailable >= this.workletFrameSize) {
            this.outputBuffer.pull(output);
        } else {
            console.warn("WARNING! AudioWorklet: Not enogh samples");
        }

        return true;
    }  
}

registerProcessor("audio-worklet-example", AudioWorkletExample);
