import {useEffect,useRef,useState} from 'react';
import {Crosshair,BoxSelect,Ban,Minus,Plus,CircleDashed} from 'lucide-react';
import {Project,Reflection,Profile,ReflectionData,DetectorGeometry} from './types';
import {Empty,Button} from './components';
import {number} from './api';

export function FrameView({project,reflections,selected,focus,onSelect,onCurate,onMask}:{project:Project;reflections:Reflection[];selected:number[];focus:number|null;onSelect:(id:number)=>void;onCurate:(ids:number[])=>void;onMask:(ids:number[])=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),wrapper=useRef<HTMLDivElement>(null);
  const [frame,setFrame]=useState(0),[mode,setMode]=useState<'point'|'box'|'mask'>('point');
  const [zoom,setZoom]=useState(1),[contrast,setContrast]=useState(1),[rings,setRings]=useState(false);
  const [image,setImage]=useState<HTMLImageElement|null>(null),[rawSize,setRawSize]=useState([1,1]);
  const [geometry,setGeometry]=useState<DetectorGeometry|null>(null),[geometryError,setGeometryError]=useState('');
  const [cursor,setCursor]=useState<number[]|null>(null),[probe,setProbe]=useState<{x:number;y:number;d:number|null}|null>(null);
  const [error,setError]=useState(''),[box,setBox]=useState<number[]|null>(null);
  const down=useRef<number[]|null>(null),transform=useRef({x:0,y:0,w:1,h:1});
  useEffect(()=>{
    if(focus===null)return;const row=reflections.find(r=>r.id===focus);
    if(row?.frame!==undefined)setFrame(Math.max(0,Math.min(project.frames.length-1,row.frame)));
  },[focus]);
  useEffect(()=>{
    let url='',active=true;setImage(null);setError('');setProbe(null);setCursor(null);
    if(!project.frames.length)return;
    fetch(`/api/projects/${project.id}/frame/${frame}.png`).then(async response=>{
      if(!response.ok){const d=await response.json();throw Error(d.error);}
      const size=[Number(response.headers.get('X-Frame-Width')),Number(response.headers.get('X-Frame-Height'))];
      const blob=await response.blob();if(!active)return;url=URL.createObjectURL(blob);const im=new Image();
      im.onload=()=>{if(active){setRawSize(size);setImage(im);}};im.src=url;
    }).catch(e=>{if(active)setError(e.message);});
    return()=>{active=false;if(url)URL.revokeObjectURL(url);};
  },[project.id,frame,project.frames.length]);
  useEffect(()=>{
    const controller=new AbortController();setGeometry(null);setGeometryError('');
    if(!project.frames.length)return;
    fetch(`/api/projects/${project.id}/view/detector?frame=${frame}`,{signal:controller.signal}).then(async response=>{
      const data=await response.json();if(!response.ok)throw Error(data.error);setGeometry(data);
    }).catch(e=>{if(e.name!=='AbortError')setGeometryError('当前几何无法计算分辨率');});
    return()=>controller.abort();
  },[project.id,project.experiments,frame]);
  useEffect(()=>{
    if(!cursor||!geometry){setProbe(null);return;}
    const controller=new AbortController();const timer=setTimeout(()=>{
      fetch(`/api/projects/${project.id}/view/detector?frame=${frame}&x=${cursor[0]}&y=${cursor[1]}`,{signal:controller.signal})
        .then(async response=>response.ok?response.json():null).then(data=>{if(data)setProbe(data);}).catch(()=>undefined);
    },110);
    return()=>{clearTimeout(timer);controller.abort();};
  },[cursor,geometry,project.id,frame]);
  useEffect(()=>{
    const element=wrapper.current,c=canvas.current;if(!element||!c||!image)return;
    const render=()=>{
      const width=element.clientWidth,height=element.clientHeight,dpr=window.devicePixelRatio,ctx=c.getContext('2d')!;
      c.width=width*dpr;c.height=height*dpr;ctx.scale(dpr,dpr);ctx.fillStyle='#222b2a';ctx.fillRect(0,0,width,height);
      const scale=Math.min(width/image.width,height/image.height)*zoom,w=image.width*scale,h=image.height*scale,x=(width-w)/2,y=(height-h)/2;
      transform.current={x,y,w,h};ctx.filter=`contrast(${contrast})`;ctx.drawImage(image,x,y,w,h);ctx.filter='none';
      if(rings&&geometry){
        ctx.save();ctx.beginPath();ctx.rect(Math.max(x,0),Math.max(y,0),Math.min(w,width),Math.min(h,height));ctx.clip();
        ctx.strokeStyle='#c5d6ce80';ctx.lineWidth=1;ctx.setLineDash([4,5]);ctx.font='12px sans-serif';
        for(const ring of geometry.rings){
          for(const segment of ring.segments){ctx.beginPath();segment.forEach((p,i)=>{const px=x+p[0]/rawSize[0]*w,py=y+p[1]/rawSize[1]*h;if(i)ctx.lineTo(px,py);else ctx.moveTo(px,py);});ctx.stroke();}
        }
        ctx.restore();
        ctx.font='12px sans-serif';const occupied:number[][]=[[10,10,190,68]],angles=[-Math.PI/2,0,Math.PI/2,-Math.PI/4,Math.PI/4,Math.PI];
        const beam=[x+geometry.beamCenter[0]/rawSize[0]*w,y+geometry.beamCenter[1]/rawSize[1]*h];
        [...geometry.rings].reverse().forEach((ring,index)=>{
          const label=`${number(ring.d,2)} Å`,tw=ctx.measureText(label).width+8,angle=angles[index%angles.length];
          const target=[beam[0]+Math.cos(angle)*w,beam[1]+Math.sin(angle)*h];
          const candidates=ring.segments.flatMap(segment=>segment.filter((_,i)=>i%4===0)).map(p=>[x+p[0]/rawSize[0]*w+4,y+p[1]/rawSize[1]*h-5])
            .filter(([px,py])=>px>Math.max(x,0)+6&&px+tw<Math.min(x+w,width)-6&&py>Math.max(y,0)+20&&py<Math.min(y+h,height)-8)
            .filter(([px,py])=>occupied.every(([ox,oy,ow,oh])=>px+tw+5<ox||px-5>ox+ow||py+7<oy||py-19>oy+oh))
            .sort((a,b)=>(a[0]-target[0])**2+(a[1]-target[1])**2-(b[0]-target[0])**2-(b[1]-target[1])**2);
          const position=candidates[0];if(!position)return;const [px,py]=position;occupied.push([px-4,py-14,tw,19]);
          ctx.fillStyle='#172523e6';ctx.fillRect(px-4,py-14,tw,19);ctx.fillStyle='#d2ddd7';ctx.fillText(label,px,py);
        });
      }
      for(const row of reflections){
        if(row.frame!==frame||row.x===undefined||row.y===undefined||(row.panel||0)!==0)continue;
        const px=x+row.x/rawSize[0]*w,py=y+row.y/rawSize[1]*h,active=selected.includes(row.id)||row.id===focus,excluded=project.excluded?.includes(row.id);
        ctx.setLineDash(excluded?[3,3]:[]);ctx.strokeStyle=excluded?'#899590':active?'#eeb66f':'#9dc1ae';ctx.lineWidth=active?2:1;
        ctx.beginPath();ctx.arc(px,py,active?6:3,0,Math.PI*2);ctx.stroke();
      }
      ctx.setLineDash([]);if(box){ctx.fillStyle='#bd7d4f22';ctx.strokeStyle='#f0b478';ctx.lineWidth=1.5;ctx.fillRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}
    };
    const observer=new ResizeObserver(render);observer.observe(element);render();return()=>observer.disconnect();
  },[image,rawSize,reflections,selected,focus,frame,zoom,contrast,box,project.excluded,rings,geometry]);
  const point=(event:React.PointerEvent)=>{const bounds=event.currentTarget.getBoundingClientRect();return [event.clientX-bounds.left,event.clientY-bounds.top];};
  const raw=(p:number[])=>{const t=transform.current;return [(p[0]-t.x)/t.w*rawSize[0],(p[1]-t.y)/t.h*rawSize[1]];};
  const currentRows=reflections.filter(r=>r.frame===frame&&(r.panel||0)===0&&r.x!==undefined&&r.y!==undefined);
  const end=(event:React.PointerEvent)=>{
    if(!down.current)return;const a=down.current,b=point(event);down.current=null;setBox(null);
    if(mode==='point'){
      const p=raw(b),closest=currentRows.reduce<Reflection|null>((best,r)=>!best||Math.hypot(r.x!-p[0],r.y!-p[1])<Math.hypot(best.x!-p[0],best.y!-p[1])?r:best,null);
      if(closest&&Math.hypot(closest.x!-p[0],closest.y!-p[1])<rawSize[0]/transform.current.w*18)onSelect(closest.id);
    }else{
      const aa=raw(a),bb=raw(b),ids=currentRows.filter(r=>r.x!>=Math.min(aa[0],bb[0])&&r.x!<=Math.max(aa[0],bb[0])&&r.y!>=Math.min(aa[1],bb[1])&&r.y!<=Math.max(aa[1],bb[1])).map(r=>r.id);
      if(mode==='mask')onMask(ids);else onCurate([...new Set([...selected,...ids])]);
    }
  };
  if(!project.frames.length)return <Empty title="尚未导入衍射帧">导入原始数据后，可在这里查看图像、选择斑点和检查屏蔽区域。</Empty>;
  return <div className="frame-view">
    <div className="frame-toolbar"><div className="segmented">
      <button aria-label="点选斑点" className={mode==='point'?'selected':''} onClick={()=>setMode('point')}><Crosshair size={19}/>点选</button>
      <button aria-label="框选斑点" className={mode==='box'?'selected':''} onClick={()=>setMode('box')}><BoxSelect size={19}/>框选</button>
      <button aria-label="屏蔽斑点" className={mode==='mask'?'selected':''} onClick={()=>setMode('mask')}><Ban size={18}/>屏蔽</button>
    </div><div className="inline">
      <button className={`icon-button ${rings?'selected':''}`} aria-label="显示分辨率环" aria-pressed={rings} title="分辨率环（Å）" disabled={!geometry} onClick={()=>setRings(v=>!v)}><CircleDashed size={19}/></button>
      <button className="icon-button" aria-label="缩小图像" onClick={()=>setZoom(z=>Math.max(.5,z-.25))}><Minus size={18}/></button>
      <button className="icon-button" aria-label="放大图像" onClick={()=>setZoom(z=>Math.min(4,z+.25))}><Plus size={18}/></button>
    </div></div>
    <div className="frame-canvas" ref={wrapper}>
      {error?<Empty title={error}/>:!image?<span className="canvas-loading">正在读取衍射图像…</span>:null}
      <canvas ref={canvas} onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);down.current=point(e);}}
        onPointerMove={e=>{const p=point(e);if(down.current&&mode!=='point')setBox([...down.current,...p]);else if(!down.current){const xy=raw(p);setCursor(xy[0]>=0&&xy[0]<rawSize[0]&&xy[1]>=0&&xy[1]<rawSize[1]?xy:null);}}}
        onPointerLeave={()=>setCursor(null)} onPointerCancel={()=>{down.current=null;setBox(null);}} onPointerUp={end}/>
      {image&&<div className="frame-resolution-readout">{probe?<><strong>{probe.d?`${number(probe.d,3)} Å`:'直射束'}</strong><span>x {number(probe.x,1)} · y {number(probe.y,1)}</span></>:<><strong>{geometry?'悬停查看分辨率':geometryError||'读取探测器几何…'}</strong><span>{currentRows.length} 个斑点{geometry?` · λ ${number(geometry.wavelength,4)} Å`:''}</span></>}</div>}
    </div>
    <div className="frame-bottom"><label>帧 {frame+1} / {project.frames.length}<input aria-label="切换衍射帧" type="range" min={0} max={project.frames.length-1} value={frame} onChange={e=>setFrame(+e.target.value)}/></label><label>对比度<input aria-label="衍射图对比度" type="range" min=".6" max="3" step=".1" value={contrast} onChange={e=>setContrast(+e.target.value)}/></label></div>
  </div>;
}

export function ProfileView({profile,onIntegrate,busy}:{profile:Profile|null;onIntegrate?:()=>void;busy?:boolean}){
  if(!profile?.observed.length)return <Empty title="选择一条反射查看像素剖面">原始衍射帧提供实际像素计数；也可打开校准像素示例检查积分。</Empty>;
  const w=680,h=270,pad={l:66,r:22,t:24,b:54};const xs=profile.x,ys=profile.observed;const minX=Math.min(...xs),maxX=Math.max(...xs);const minY=Math.min(0,...ys),maxY=Math.max(...ys)*1.05||1;
  const x=(n:number)=>pad.l+(n-minX)/(maxX-minX||1)*(w-pad.l-pad.r);const y=(n:number)=>h-pad.b-(n-minY)/(maxY-minY||1)*(h-pad.t-pad.b);
  const d=ys.map((v,i)=>`${i?'L':'M'}${x(xs[i])},${y(v)}`).join(' ');
  return <div className="profile-view"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={profile.label}>{[0,.25,.5,.75,1].map(t=>{const val=minY+(maxY-minY)*t;return <g key={t}><line x1={pad.l} y1={y(val)} x2={w-pad.r} y2={y(val)} stroke="#d7dcd8"/><text x={pad.l-10} y={y(val)+5} textAnchor="end">{number(val,0)}</text></g>;})}<path d={`${d} L${x(xs[xs.length-1])},${y(minY)} L${x(xs[0])},${y(minY)}Z`} fill="#b3542c15"/><path d={d} fill="none" stroke="#b3542c" strokeWidth="2.8"/>{ys.map((v,i)=><circle key={i} cx={x(xs[i])} cy={y(v)} r={ys.length<100?3:0} fill="#b3542c"/>)}<text x={pad.l} y={h-16}>{number(minX,0)}</text><text x={w-pad.r} y={h-16} textAnchor="end">{number(maxX,0)}</text><text x={w/2} y={h-12} textAnchor="middle">像素位置</text></svg><p className="plot-caption">{profile.label}</p>{onIntegrate&&<Button variant="secondary" busy={busy} onClick={onIntegrate}>运行校准像素积分</Button>}</div>;
}

export function ReflectionTable({data,focus,selected,onFocus,onCurate}:{data:ReflectionData|null;focus:number|null;selected:number[];onFocus:(id:number)=>void;onCurate:(ids:number[])=>void}){
  const [query,setQuery]=useState('');const [only,setOnly]=useState(false);const set=new Set(selected);
  const hasPixels=data?.reflections.some(r=>r.x!==undefined);
  const rows=(data?.reflections||[]).filter(r=>(!only||set.has(r.id))&&(!query||r.hkl.join(' ').includes(query)||String(r.id).includes(query))).slice(0,150);
  return <section className="reflection-table"><div className="section-heading"><h3>反射与选点</h3><div className="inline"><input className="search-field" aria-label="搜索 HKL" placeholder="搜索 h k l" value={query} onChange={e=>setQuery(e.target.value)}/><label className="checkbox"><input type="checkbox" checked={only} onChange={e=>setOnly(e.target.checked)}/>仅已选</label></div></div><div className="table-scroll"><table><thead><tr><th>选择</th><th>h k l</th>{data?.reciprocalUnit==='Å⁻¹'&&<th>分辨率 / Å</th>}{hasPixels&&<><th>帧</th><th>像素 x / y</th></>}<th>强度 I</th><th>σ(I)</th><th>I / σ</th><th>晶体模型</th></tr></thead><tbody>{rows.map(r=><tr key={r.id} className={focus===r.id?'focused':''} onClick={()=>onFocus(r.id)}><td><input type="checkbox" aria-label={`选择反射 ${r.id}`} checked={set.has(r.id)} onClick={e=>e.stopPropagation()} onChange={()=>onCurate(set.has(r.id)?selected.filter(id=>id!==r.id):[...selected,r.id])}/></td><td className="mono">{r.indexed===false?'未指标化':r.hkl.join('  ')}</td>{data?.reciprocalUnit==='Å⁻¹'&&<td>{number(r.d,3)}</td>}{hasPixels&&<><td>{r.frame===undefined?'—':r.frame+1}</td><td className="mono">{number(r.x,1)} / {number(r.y,1)}</td></>}<td>{number(r.intensity)}</td><td>{number(r.sigma)}</td><td>{number(r.intensity!==null&&r.sigma?r.intensity/r.sigma:null)}</td><td>{r.domain===undefined?'—':r.domain+1}</td></tr>)}</tbody></table>{!rows.length&&<p className="table-empty">尚无符合条件的反射。运行找点或导入反射文件后可查看。</p>}</div><div className="table-note"><span>共 {data?.total||0} 条反射 · 已选 {selected.length} 条</span><span>负强度保留 · 表格显示前 150 条匹配结果</span>{!!selected.length&&<button className="text-button" onClick={()=>onCurate([])}>清除选择</button>}</div></section>;
}
