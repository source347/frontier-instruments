(()=>{"use strict";
const $=id=>document.getElementById(id);
const fmtMs=n=>n<1?(n*1000).toFixed(1)+" µs":n.toFixed(2)+" ms";
const fmtNum=n=>Number(n).toLocaleString("en-US");
let device=null,adapter=null,limits={},gpuResult=null,cpuResult=null,runSerial=0;
const shaderAdd=`
struct Data { values: array<f32> };
@group(0) @binding(0) var<storage, read> a: Data;
@group(0) @binding(1) var<storage, read> b: Data;
@group(0) @binding(2) var<storage, read_write> output: Data;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i = gid.x;
 if (i >= arrayLength(&output.values)) { return; }
 output.values[i] = a.values[i] + b.values[i];
}`;
const shaderReduce=`
struct Data { values: array<f32> };
@group(0) @binding(0) var<storage, read> inputData: Data;
@group(0) @binding(1) var<storage, read_write> partials: Data;
var<workgroup> scratch: array<f32, 256>;
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
 let i = gid.x;
 var value = 0.0;
 if (i < arrayLength(&inputData.values)) { value = inputData.values[i]; }
 scratch[lid] = value;
 workgroupBarrier();
 var stride = WG_SIZE / 2u;
 loop {
  if (stride == 0u) { break; }
  if (lid < stride) { scratch[lid] = scratch[lid] + scratch[lid + stride]; }
  workgroupBarrier();
  stride = stride / 2u;
 }
 if (lid == 0u) { partials.values[wid.x] = scratch[0]; }
}`;
function config(){
const n=Number($("elements").value),wg=Number($("workgroup").value),groups=Math.ceil(n/wg),op=$("operation").value;
$("dispatch-summary").textContent=fmtNum(n)+" elements · "+wg+" threads/workgroup · "+fmtNum(groups)+" workgroups";
$("groups-count").textContent=fmtNum(groups);$("threads-count").textContent=fmtNum(wg);$("invocations-count").textContent=fmtNum(groups*wg);
$("shader-preview").textContent=op==="add"?shaderAdd.replace("WG_SIZE",String(wg)).replaceAll("&","&").replaceAll("<","<"):shaderReduce.replace("WG_SIZE",String(wg));
return{n,wg,groups,op};
}
["elements","workgroup","operation"].forEach(id=>$(id).addEventListener("change",config));
function updateBars(){
const g=gpuResult?.ms,c=cpuResult?.ms,max=Math.max(g||0,c||0)||1;
$("gpu-bar").style.width=g===undefined?"0%":(g/max*100)+"%";$("cpu-bar").style.width=c===undefined?"0%":(c/max*100)+"%";
$("gpu-bar-label").textContent=g===undefined?"—":fmtMs(g);$("cpu-bar-label").textContent=c===undefined?"—":fmtMs(c);
}
function setStatus(s){$("run-status").textContent=s}
async function init(){
if(!("gpu" in navigator)){ $("device-tag").textContent="UNAVAILABLE";$("api-status").textContent="WebGPU not exposed";$("adapter-name").textContent="No WebGPU API";$("device-note").textContent="Use a current browser with WebGPU enabled, over HTTPS or localhost. The CPU baseline remains available.";return}
try{
adapter=await navigator.gpu.requestAdapter();
if(!adapter)throw new Error("No compatible GPU adapter was returned by the browser.");
limits=adapter.limits;
device=await adapter.requestDevice();
device.lost.then(info=>{device=null;$("run-gpu").disabled=true;$("device-tag").textContent="DEVICE LOST";setStatus("GPU device lost: "+(info.message||info.reason));});
$("device-tag").textContent="READY";$("api-status").textContent="Available";$("adapter-name").textContent="Adapter acquired (browser privacy may hide vendor details)";$("workgroup-limit").textContent=String(limits.maxComputeWorkgroupSizeX||"—")+" threads / axis";
$("device-note").textContent="A WebGPU device is ready. The page requests only the default device features.";
$("run-gpu").disabled=false;setStatus("Ready. Choose a workload and run it on the GPU.");
}catch(err){$("device-tag").textContent="UNAVAILABLE";$("api-status").textContent="Initialization failed";$("adapter-name").textContent="No device";$("device-note").textContent=err.message+" CPU execution is still available."}
}
function makeInputs(n,op){
const a=new Float32Array(n),b=op==="add"?new Float32Array(n):null;
for(let i=0;i<n;i++){a[i]=((i%997)-498)/997;if(b)b[i]=((i%313)-156)/313}
return{a,b};
}
function cpuCompute(data,op){
const t0=performance.now();let result;
if(op==="add"){result=new Float32Array(data.a.length);for(let i=0;i<result.length;i++)result[i]=data.a[i]+data.b[i]}
else{let sum=0;for(let i=0;i<data.a.length;i++)sum+=data.a[i];result=sum}
return{result,ms:performance.now()-t0};
}
function reference(data,op){if(op==="add")return data.a[0]+data.b[0];let sum=0;for(let i=0;i<data.a.length;i++)sum+=data.a[i];return sum}
function buffer(size,usage){return device.createBuffer({size:Math.max(4,Math.ceil(size/4)*4),usage})}
async function runGpu(){
if(!device){setStatus("No active GPU device. Use the CPU baseline or reload in a WebGPU-capable browser.");return}
const serial=++runSerial,{n,wg,groups,op}=config(),data=makeInputs(n,op),bytes=n*4;
if(bytes>device.limits.maxStorageBufferBindingSize){setStatus("Selected workload exceeds this device's storage-buffer binding limit.");return}
$("run-gpu").disabled=true;setStatus("Compiling pipeline and dispatching "+fmtNum(n)+" elements…");
let aBuf,bBuf,outBuf,readBuf,partialBuf;
try{
const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
aBuf=buffer(bytes,usage);device.queue.writeBuffer(aBuf,0,data.a);
let pipeline,bindGroup,dispatchOut;
if(op==="add"){
bBuf=buffer(bytes,usage);device.queue.writeBuffer(bBuf,0,data.b);
outBuf=buffer(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
const module=device.createShaderModule({code:shaderAdd});
pipeline=device.createComputePipeline({layout:"auto",compute:{module,entryPoint:"main",constants:{WG_SIZE:wg}}});
bindGroup=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:aBuf}},{binding:1,resource:{buffer:bBuf}},{binding:2,resource:{buffer:outBuf}}]});
dispatchOut=outBuf;
}else{
const partialBytes=groups*4;
partialBuf=buffer(partialBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
const module=device.createShaderModule({code:shaderReduce});
pipeline=device.createComputePipeline({layout:"auto",compute:{module,entryPoint:"main",constants:{WG_SIZE:wg}}});
bindGroup=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:aBuf}},{binding:1,resource:{buffer:partialBuf}}]});
dispatchOut=partialBuf;
}
readBuf=buffer(op==="add"?bytes:groups*4,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
pass.setPipeline(pipeline);pass.setBindGroup(0,bindGroup);pass.dispatchWorkgroups(groups);pass.end();
encoder.copyBufferToBuffer(dispatchOut,0,readBuf,0,op==="add"?bytes:groups*4);
const t0=performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();await readBuf.mapAsync(GPUMapMode.READ);const elapsed=performance.now()-t0;
const copied=readBuf.getMappedRange().slice(0),view=new Float32Array(copied);let sample,valid;
if(op==="add"){sample=Array.from(view.slice(0,8));valid=Math.abs(view[0]-reference(data,op))<1e-6}
else{let sum=0;for(const x of view)sum+=x;sample=[sum];const ref=reference(data,op);valid=Math.abs(sum-ref)<Math.max(1e-3,Math.abs(ref)*1e-4)}
readBuf.unmap();if(serial!==runSerial)return;
gpuResult={ms:elapsed,sample};$("gpu-time").textContent=fmtMs(elapsed);$("gpu-throughput").textContent=(n/(elapsed/1000)/1e6).toFixed(2)+" M elements/s (effective)";
$("validation").textContent=valid?"PASS":"CHECK";$("validation-note").textContent=valid?"GPU output matches CPU reference within tolerance.":"Output differs from CPU reference; inspect precision and device behavior.";
$("output-label").textContent=op==="add"?"FIRST 8 OUTPUT ELEMENTS":"REDUCTION RESULT";
$("output-preview").textContent=op==="add"?sample.map((v,i)=>"["+i+"] "+v.toFixed(6)).join("\n"):"Σ Aᵢ = "+sample[0].toFixed(6)+"\nPartial sums: "+groups+" workgroups";
updateBars();setStatus("GPU workload completed. Timing includes command submission, execution completion, and output readback.");
}catch(err){if(serial===runSerial){$("validation").textContent="ERROR";$("validation-note").textContent=err.message;setStatus("GPU execution failed: "+err.message)}}
finally{[aBuf,bBuf,outBuf,readBuf,partialBuf].forEach(x=>{if(x)x.destroy()});if(serial===runSerial)$("run-gpu").disabled=!device}
}
function runCpu(){
const {n,op}=config(),data=makeInputs(n,op),result=cpuCompute(data,op);
cpuResult={ms:result.ms};$("cpu-time").textContent=fmtMs(result.ms);$("cpu-throughput").textContent=(n/(result.ms/1000||1e-9)/1e6).toFixed(2)+" M elements/s";
$("output-label").textContent=op==="add"?"CPU FIRST 8 OUTPUT ELEMENTS":"CPU REDUCTION RESULT";
$("output-preview").textContent=op==="add"?Array.from(result.result.slice(0,8)).map((v,i)=>"["+i+"] "+v.toFixed(6)).join("\n"):"Σ Aᵢ = "+result.result.toFixed(6);
if(gpuResult){const expected=reference(data,op),actual=op==="add"?result.result[0]:result.result;const ok=Math.abs(expected-actual)<1e-6;$("validation").textContent=ok?"PASS":"CHECK"}
updateBars();setStatus("CPU baseline completed using a JavaScript Float32Array loop. Run the GPU workload with the same settings for comparison.");
}
$("run-gpu").addEventListener("click",runGpu);$("run-cpu").addEventListener("click",runCpu);
$("reset").addEventListener("click",()=>{gpuResult=cpuResult=null;$("gpu-time").textContent=$("cpu-time").textContent=$("validation").textContent="—";$("gpu-throughput").textContent="No GPU run yet";$("cpu-throughput").textContent="No CPU run yet";$("validation-note").textContent="Results are checked against a CPU reference.";$("output-preview").textContent="Run a workload to inspect sample outputs.";$("gpu-bar").style.width=$("cpu-bar").style.width="0%";$("gpu-bar-label").textContent=$("cpu-bar-label").textContent="—";setStatus("Results cleared.")});
config();init();
})();