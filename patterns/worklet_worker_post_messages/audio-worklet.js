import {RingBuffer} from 'ringbuffer.js';

class NoiseCancellationProcessor extends AudioWorkletProcessor {
    inputBuffer;
    outputBuffer;
    portToWorker;

    denoiseFrameSize = 480;
    workletFrameSize = 128;

    processingDenoiseFramesCount = 0;
    processingDenoiseFramesCountLimit = 300;xwi 

    denoiseLatencyFramesCount = 4;
    readyToReturnOutput = false;

    currentFrameAfterInit;
    
    constructor(options) {
      super();

      const ringBufferLength = leastCommonMultiple(this.workletFrameSize, this.denoiseFrameSize) * 2;
      this.inputBuffer = new RingBuffer(ringBufferLength, 1);
      this.outputBuffer = new RingBuffer(ringBufferLength, 1);

      this.currentFrameAfterInit = 0;

      this.port.onmessage = (e) => {
          if (e.data instanceof MessagePort) {
              this.portToWorker = e.data;
              this.portToWorker.onmessage = (event) => {
                  // console.log('AudioWorklet: Got message from Worker...');
                  const { output } = event.data;
                  const outputArray = new Float32Array(output);
                  this.outputBuffer.push([outputArray]);
                  this.processingDenoiseFramesCount--;
              }
          };
      };
    }

    process(inputs, outputs) {
        const [firstInSource] = inputs;
        const [firstOutSource] = outputs;
        const [inData] = firstInSource;
        const [outData] = firstOutSource;

        if (!this.portToWorker) {
          outData.set(inData, 0);
          return true;
        }
    
        // as rest of worklet will crash otherwise
        if (!inData) {
            this.port.close();
            return false;
        }

        this.inputBuffer.push([inData]);

        if (this.inputBuffer.samplesAvailable > this.denoiseFrameSize) {
            const [input] = this.inputBuffer.pullArraySliceBuffer(this.denoiseFrameSize);

            // console.log('AudioWorklet: Pushing to Worker');
            this.portToWorker.postMessage({ input }, [input]);

            this.processingDenoiseFramesCount++;

            if (
                this.processingDenoiseFramesCount > this.processingDenoiseFramesCountLimit &&
                this.outputBuffer.samplesAvailable < this.workletFrameSize
            ) {
                throw new Error(
                  `Limit ${this.processingDenoiseFramesCountLimit} of frames from noise cancellation worker is reached. Current frames count ${this.processingDenoiseFramesCount}. Now ${this.outputBuffer.samplesAvailable} samples available to read`,
                );
            }
        }

        // console.log(this.outputBuffer.samplesAvailable);
        console.timeStamp('inp: ' + this.inputBuffer.samplesAvailable + ' / out: ' + this.outputBuffer.samplesAvailable);

        if (!this.readyToReturnOutput) {
            this.readyToReturnOutput = 
            this.outputBuffer.samplesAvailable >= this.denoiseLatencyFramesCount * this.denoiseFrameSize;

            console.log('AudioWorklet: Waiting for padding');

            return true;
        } else if (this.outputBuffer.samplesAvailable >= this.workletFrameSize) {
            const [output] = this.outputBuffer.pullSubarray(this.workletFrameSize);

            outData.set(output, 0);
        } else {
            // Review it later
            console.timeStamp(`NOT ENOUGH SAMPLES`);
            console.warn("WARNING! AudioWorklet: Not enogh samples");
            const currentReadIdx = this.outputBuffer.readIndex;
            const [output] = this.inputBuffer.getSubarray(currentReadIdx, this.workletFrameSize);
            this.outputBuffer.skip(this.workletFrameSize);
            outData.set(output, 0);
        }

        return true;
    }  
}

registerProcessor("audio-worklet-example", NoiseCancellationProcessor);
