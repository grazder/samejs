const exports = {};

class RandomAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    // Staging buffer to interleave the audio data.
    const rawSab = options.processorOptions.rawSab;
    const denoisedSab = options.processorOptions.denoisedSab;
    this.planarBuffer = new Float32Array(128);
    this._audio_writer = new AudioWriter(new RingBuffer(rawSab, Float32Array));
    this._audio_reader = new AudioReader(new RingBuffer(denoisedSab, Float32Array));
  }

  process(inputList, outputList, parameters) {
    const sourceLimit = Math.min(inputList.length, outputList.length);

    // interleave and store in the queue
    if (this._audio_writer.enqueue(inputList[0][0]) !== 128) {
        console.log("audioworklet underrun: the worker doesn't dequeue fast enough!");
    }

    if (this._audio_reader.available_read() >= 128){
        const samples_read = this._audio_reader.dequeue(this.planarBuffer);
        if (samples_read < 128){
          console.log(`ERROR! this._audio_reader in audioworklet read ${samples_read}`)

          return false
        }

        for (let inputNum = 0; inputNum < sourceLimit; inputNum++) {
          const input = inputList[inputNum];
          const output = outputList[inputNum];
    
          const channelCount = Math.min(input.length, output.length);
    
          for (let channelNum = 0; channelNum < channelCount; channelNum++) {
            this.planarBuffer.forEach((sample, i) => {
              output[channelNum][i] = sample;
            });
          }
        };
    } else {
        console.log(`audioworklet underrun: the worker doesn't enqueue fast enough! available only - ${this._audio_reader.available_read()}`);
    }

    return true;
  }  
}

registerProcessor("random-audio-processor", RandomAudioProcessor);
