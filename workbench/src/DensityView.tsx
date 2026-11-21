import {useEffect,useMemo,useState} from 'react';
import {Download,RefreshCw,Upload} from 'lucide-react';
import {Crystal,Density,Project} from './types';
import {Button,Empty} from './components';
import {number} from './api';
import ThreeView from './ThreeView';

export default function DensityView({project,crystal,busy,onGenerate,onImport}:{project:Project;crystal:Crystal|null;busy:boolean;onGenerate:()=>void;onImport:()=>void}){
  const [map,setMap]=useState(''),[sigma,setSigma]=useState(3),[draft,setDraft]=useState(3);
  const [data,setData]=useState<Density|null>(null),[error,setError]=useState(''),[loading,setLoading]=useState(true);
  const [atoms,setAtoms]=useState(true),[wireframe,setWireframe]=useState(true),[opacity,setOpacity]=useState(.55);
  const cellStructure=useMemo(()=>{if(!crystal?.atoms.length)return null;const indices=new Map<number,number>();const atoms=crystal.atoms.filter((atom,index)=>{if(atom.periodicImage)return false;indices.set(index,indices.size);return true;});const bonds=crystal.bonds.filter(([a,b])=>indices.has(a)&&indices.has(b)).map(([a,b])=>[indices.get(a)!,indices.get(b)!]);return {...crystal,atoms,bonds};},[crystal]);
  useEffect(()=>{const timer=setTimeout(()=>setSigma(draft),180);return()=>clearTimeout(timer);},[draft]);
  useEffect(()=>{
    const controller=new AbortController();setLoading(true);setError('');
    fetch(`/api/projects/${project.id}/view/density?map=${encodeURIComponent(map)}&sigma=${sigma}`,{signal:controller.signal})
      .then(async response=>{const result=await response.json();if(!response.ok)throw Error(result.error);setData(result);})
      .catch(e=>{if(e.name!=='AbortError')setError(e.message);}).finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return()=>controller.abort();
  },[project.id,project.surface,project.files.length,map,sigma]);
  const chosen=data?.availableMaps?.find(item=>item.id===data.mapId),ready=!!data?.availableMaps?.length;
  const canGenerate=!!(project.structure&&project.reflections);
  const shelx=!!project.structure&&/\.(ins|res)$/i.test(project.structure);
  return <div className="density-view" aria-label="电子密度图查看器">
    {ready&&<div className="density-toolbar">
      <label className="density-map-picker">密度图<select aria-label="密度图类型" value={map||data?.mapId||''} onChange={e=>{setMap(e.target.value);setAtoms(e.target.value!=='patterson');const level=e.target.value==='difference'?3:1.5;setDraft(level);setSigma(level);}}>{data?.availableMaps?.map(item=><option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
      <label className="density-contour">等值面 <strong>{chosen?.signed?'±':''}{number(draft,2)}σ</strong><input aria-label="密度等值面 σ" type="range" min=".5" max="8" step=".25" value={draft} onChange={e=>setDraft(+e.target.value)}/></label>
    </div>}
    {error&&<p className="density-error" role="alert">{error}</p>}
    <div className="density-canvas" aria-busy={loading}>
      {ready?<><ThreeView crystal={atoms?cellStructure:null} density={data} showDensity densityOpacity={opacity} densityWireframe={wireframe} showCell kind="density"/>
        {!data?.surfaces.length&&!loading&&<div className="density-empty-contour">当前阈值没有等值面，请降低 σ。</div>}
        <div className="density-legend"><span><i/>正密度</span>{chosen?.signed&&<span><i className="negative"/>负密度</span>}<span>{data?.type}</span></div>
      </>:loading?<div className="canvas-loading">正在读取电子密度图…</div>:<Empty title="电子密度图"><span>导入 CCP4 / MAP，或用结构与 HKL 计算密度图。</span><span className="density-empty-actions"><Button disabled={!canGenerate||busy} busy={busy} onClick={onGenerate}>{shelx?'使用 SHELXL 计算':'计算密度图'}</Button><Button variant="secondary" onClick={onImport}><Upload size={17}/>导入密度图</Button></span>{shelx&&<span className="caption">使用独立的零循环副本生成结构因子，原文件保留。</span>}{!canGenerate&&<span className="caption">计算需要结构模型与 HKL 反射数据。</span>}</Empty>}
      {ready&&loading&&<span className="density-loading">更新等值面…</span>}
    </div>
    {ready&&<div className="density-options">
      <label className="checkbox"><input type="checkbox" checked={atoms} disabled={!crystal?.atoms.length} onChange={e=>setAtoms(e.target.checked)}/>叠加结构</label>
      <label className="checkbox"><input type="checkbox" checked={wireframe} onChange={e=>setWireframe(e.target.checked)}/>网格等值面</label>
      <details><summary>更多</summary><div className="density-more">
        <label>透明度<input aria-label="密度图透明度" type="range" min=".1" max="1" step=".05" value={opacity} onChange={e=>setOpacity(+e.target.value)}/></label>
        {data?.statistics&&<><p>网格 {data.statistics.grid.join(' × ')}<br/>σ = {number(data.statistics.sigma,4)} {data.statistics.units==='e/angstrom^3'?'e/Å³':'原图单位'}</p><p>σ 按本图平均值和标准差归一化。</p></>}
        {chosen?.download&&<a className="text-button" href={`/api/projects/${project.id}/files/${chosen.download.split('/').map(encodeURIComponent).join('/')}`} download><Download size={15}/>下载 CCP4</a>}
        <button className="text-button" disabled={!canGenerate||busy} onClick={onGenerate}><RefreshCw size={15}/>重新计算</button>
        <button className="text-button" onClick={onImport}><Upload size={15}/>导入 CCP4 / MAP</button>
      </div></details>
    </div>}
  </div>;
}
