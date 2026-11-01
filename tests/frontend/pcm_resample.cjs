// Worklet captures actual hardware sample rate and emits 16kHz mono PCM samples.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
for(const rate of [44100,48000]) {
  let Processor; const output=[];
  class Worklet { constructor(){this.port={postMessage:value=>output.push(...value)};} }
  const context={AudioWorkletProcessor:Worklet,sampleRate:rate,registerProcessor:(_,p)=>Processor=p,Float32Array};
  vm.createContext(context); vm.runInContext(fs.readFileSync('src/ciel/web_assets/pcm-worklet.js','utf8'),context);
  const processor=new Processor();
  for(let offset=0;offset<rate;offset+=128) processor.process([[new Float32Array(Math.min(128,rate-offset)).fill(.5)]]);
  assert(Math.abs(output.length-16000)<=1, 'One second must produce 16kHz audio at '+rate);
  assert(output.every(value=>Number.isFinite(value)&&Math.abs(value-.5)<.001));
}
console.log('44.1 and 48kHz microphone resampling passed.');
