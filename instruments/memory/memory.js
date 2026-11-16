(()=>{"use strict";
const $=id=>document.getElementById(id);
const initial=[
{id:1,text:"Prefers concise technical reports with explicit assumptions.",type:"preference",importance:5,strength:1,day:0},
{id:2,text:"Project uses a static dependency-free GitHub Pages architecture.",type:"fact",importance:4,strength:1,day:1},
{id:3,text:"Reviewed transformer attention and grouped-query attention.",type:"event",importance:3,strength:1,day:3},
{id:4,text:"For code changes, run browser checks before publishing.",type:"procedure",importance:4,strength:1,day:5},
{id:5,text:"Prefers technical terminology over simplified explanations.",type:"preference",importance:5,strength:1,day:8},
{id:6,text:"Compared KV cache estimates at multiple precisions.",type:"event",importance:2,strength:1,day:12}
];
let memories=[],day=0,nextId=7,lastResults=[];
const clone=()=>initial.map(x=>({...x}));
const tokens=s=>(s.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(x=>x.length>2);
const escape=s=>s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const age=m=>Math.max(0,day-m.day);
function renderStore(){
$("store-count").textContent=memories.length+" RECORDS";
$("clock").textContent="Simulation day "+day;
$("memory-rows").innerHTML=memories.length?memories.map(m=>'<tr><td>'+escape(m.text)+'</td><td><span class="type-pill">'+escape(m.type)+'</span></td><td>'+m.importance+'/5</td><td><div class="strength"><span>'+Math.round(m.strength*100)+'%</span><span class="strength-track"><i style="width:'+Math.round(m.strength*100)+'%"></i></span></div></td><td>'+age(m)+'d</td><td><button class="delete-memory" data-id="'+m.id+'" aria-label="Delete memory">×</button></td></tr>').join(""):'<tr><td colspan="6">No stored memories.</td></tr>';
$("memory-rows").querySelectorAll("[data-id]").forEach(b=>b.addEventListener("click",()=>{memories=memories.filter(m=>m.id!==Number(b.dataset.id));renderStore();retrieve()}));
}
function rank(query){
const q=new Set(tokens(query));
return memories.map(m=>{
const words=tokens(m.text),unique=new Set(words);
const overlap=[...q].filter(w=>unique.has(w)).length;
const lexical=q.size?overlap/q.size:0;
const recency=Math.exp(-age(m)/45);
const strength=m.strength;
const score=(.58*lexical+.24*(m.importance/5)+.18*recency)*strength;
return {...m,score,overlap};
}).filter(m=>m.strength>=.08||$("include-decayed").checked).sort((a,b)=>b.score-a.score);
}
function retrieve(){
const query=$("query").value.trim(),budget=Number($("budget").value);
const ranked=rank(query),chosen=ranked.slice(0,budget);lastResults=chosen;
$("budget-out").textContent=budget;
$("retrieval-results").innerHTML=chosen.length?chosen.map(m=>'<article class="result-card"><p>'+escape(m.text)+'</p><span class="score">'+(m.score*100).toFixed(1)+'</span><small>'+m.type+' · importance '+m.importance+'/5 · '+age(m)+' days old · '+m.overlap+' query terms matched</small><div class="score-track"><i style="width:'+Math.round(m.score*100)+'%"></i></div></article>').join(""):'<div class="empty">No memories matched the current filters.</div>';
const count=chosen.reduce((n,m)=>n+tokens(m.text).length,0);
$("context-tokens").textContent=count;
$("context-summary").textContent=chosen.length+" memories selected from "+memories.length+" stored records. Token count is a whitespace/word estimate, not a tokenizer measurement.";
$("context-fill").style.width=Math.min(100,count/512*100)+"%";
}
function decay(){
memories=memories.map(m=>({...m,strength:Math.max(0.02,m.strength*Math.pow(.985,Math.max(0,age(m)-m.lastDecayAge||0)))}));
}
$("advance").addEventListener("click",()=>{
const step=Number($("time-step").value);day+=step;
memories=memories.map(m=>({...m,strength:Math.max(.01,m.strength*Math.pow(.985,step))}));
renderStore();retrieve();
});
$("reset").addEventListener("click",()=>{memories=clone();day=0;nextId=7;$("consolidation-log").innerHTML="";$("consolidation-status").textContent="Simulation reset.";renderStore();retrieve()});
$("importance").addEventListener("input",()=>{$("importance-out").textContent=$("importance").value});
$("add-form").addEventListener("submit",e=>{
e.preventDefault();const text=$("memory-text").value.trim();if(!text)return;
memories.push({id:nextId++,text,type:$("memory-type").value,importance:Number($("importance").value),strength:1,day});
$("memory-text").value="";renderStore();retrieve();
});
$("retrieve").addEventListener("click",retrieve);
$("query").addEventListener("keydown",e=>{if(e.key==="Enter")retrieve()});
$("budget").addEventListener("input",retrieve);
$("include-decayed").addEventListener("change",retrieve);
function similarity(a,b){const A=new Set(tokens(a)),B=new Set(tokens(b));if(!A.size||!B.size)return 0;let n=0;for(const x of A)if(B.has(x))n++;return n/(A.size+B.size-n)}
$("consolidate").addEventListener("click",()=>{
const groups=[],used=new Set();
for(let i=0;i<memories.length;i++){if(used.has(memories[i].id))continue;const group=[memories[i]];for(let j=i+1;j<memories.length;j++){if(!used.has(memories[j].id)&&similarity(memories[i].text,memories[j].text)>=.28){group.push(memories[j]);used.add(memories[j].id)}}if(group.length>1){group.forEach(m=>used.add(m.id));groups.push(group)}}
if(!groups.length){$("consolidation-status").textContent="No overlapping records met the similarity threshold.";return}
const log=[];
for(const group of groups){
const combined=[...new Set(group.flatMap(m=>tokens(m.text)))].slice(0,24).join(" ");
const summary="Consolidated memory: "+combined+".";
const sourceIds=group.map(m=>m.id);
const importance=Math.max(...group.map(m=>m.importance));
memories=memories.map(m=>sourceIds.includes(m.id)?({...m,strength:m.strength*.45}):m);
memories.push({id:nextId++,text:summary,type:"fact",importance,strength:1,day});
log.push("Merged "+group.length+" records (IDs "+sourceIds.join(", ")+") into: "+summary);
}
$("consolidation-status").textContent=groups.length+" group(s) consolidated; source records retained at reduced strength.";
$("consolidation-log").innerHTML=log.map(x=>'<div class="log-entry">'+escape(x)+'</div>').join("");
renderStore();retrieve();
});
memories=clone();renderStore();retrieve();
})();