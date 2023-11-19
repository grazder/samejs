class NoiseCancellationProcessor extends AudioWorkletProcessor {
    inputBuffer;
    outputBuffer;
    portToWorker;

    denoiseFrameSize = 480;
    workletFrameSize = 128;

    processingDenoiseFramesCount = 0;
    processingDenoiseFramesCountLimit = 300;

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

/**
 * Calculate least common multiple using gcd.
 *
 * @param {number} num1 - First number.
 * @param {number} num2 - Second number.
 * @returns {number}
 */
export function leastCommonMultiple(num1, num2) {
  const number1 = num1;
  const number2 = num2;

  const gcd = greatestCommonDivisor(number1, number2);

  return (number1 * number2) / gcd;
}

/**
 * Compute the greatest common divisor using Euclid's algorithm.
 *
 * @param {number} num1 - First number.
 * @param {number} num2 - Second number.
 * @returns {number}
 */
export function greatestCommonDivisor(num1, num2) {
  let number1 = num1;
  let number2 = num2;

  while (number1 !== number2) {
    if (number1 > number2) {
      number1 = number1 - number2;
    } else {
      number2 = number2 - number1;
    }
  }

  return number2;
}

class RingBuffer {
  _readIndex;
  _writeIndex;
  _samplesAvailable;
  _channelCount;
  _length;
  _channelData;

  /**
   * @constructor
   * @param  {number} length Buffer length in samples.
   * @param  {number} channelCount Buffer channel count.
   */
  constructor(length, channelCount) {
    this._readIndex = 0;
    this._writeIndex = 0;
    this._samplesAvailable = 0;

    this._channelCount = channelCount;
    this._length = length;
    this._channelData = [];
    for (let i = 0; i < this._channelCount; ++i) {
      this._channelData[i] = new Float32Array(length);
    }
  }

  /**
   * Getter for Available samples in buffer.
   *
   * @return {number} Available samples in buffer.
   */
  get samplesAvailable() {
    return this._samplesAvailable;
  }

  get readIndex() {
    return this._readIndex;
  }
  /**
   * Push a sequence of Float32Arrays to buffer.
   *
   * @param  {array} arraySequence A sequence of Float32Arrays.
   */
  push(arraySequence) {
    const sourceLength = arraySequence[0].length;
    const nextWriteIndex = this._writeIndex + sourceLength;

    arraySequence.forEach((channel, channelIdx) => {
      const channelSubarray = this._channelData[channelIdx].subarray(this._writeIndex, nextWriteIndex);
      channel.forEach((sample, sampleIdx) => {
        channelSubarray[sampleIdx] = sample;
      });
    });

    this._writeIndex = nextWriteIndex === this._length ? 0 : nextWriteIndex;

    this._samplesAvailable = Math.min(this._samplesAvailable + sourceLength, this._length);

    if (this._samplesAvailable > this._length){
      console.log('WTF push', this._samplesAvailable, this._length);
    }
  }

  pullArraySliceBuffer(arraySliceLength) {
    const arrayStart = this._readIndex;
    const arrayEnd = this._readIndex + arraySliceLength;

    const result = this._channelData.map((channel) =>
      channel.buffer.slice(arrayStart * channel.BYTES_PER_ELEMENT, arrayEnd * channel.BYTES_PER_ELEMENT),
    );


    this._readIndex = (this._readIndex + arraySliceLength) % this._length;
    this._samplesAvailable = Math.max(this._samplesAvailable - arraySliceLength, 0);

    if (this._samplesAvailable < 0){
      console.log('WTF pullArraySliceBuffer', this._samplesAvailable, this._length);
    }

    return result;
  }

  pullSubarray(subarrayLength) {
    const arrayStart = this._readIndex;
    const arrayEnd = this._readIndex + subarrayLength;

    const result = this._channelData.map((channel) => channel.subarray(arrayStart, arrayEnd));

    if (arrayEnd > this._length){
      console.log('WTF pullSubarray', arrayStart, arrayEnd, this._length);
    }

    this._readIndex = arrayEnd === this._length ? 0 : arrayEnd;
    this._samplesAvailable = Math.max(this._samplesAvailable - subarrayLength, 0);

    if (this._samplesAvailable < 0){
      console.log('WTF pullSubarray', this._samplesAvailable, this._length);
    }
    return result;
  }

  skip(skipLength) {
    const nextReadIdx = this._readIndex + skipLength;
    this._readIndex = nextReadIdx === this._length ? 0 : nextReadIdx;
    this._samplesAvailable -= skipLength;
  }

  getSubarray(start, subarrayLength) {
    return this._channelData.map((channel) => channel.subarray(start, start + subarrayLength));
  }
}


registerProcessor("audio-worklet-example", NoiseCancellationProcessor);
