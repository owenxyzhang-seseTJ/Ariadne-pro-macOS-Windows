import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import {RotateCcw,Rotate3D,Box} from 'lucide-react';
import {Crystal,Reflection,Density} from './types';

type Props={glb?:string;poster?:string;cell?:number[][];crystal?:Crystal|null;reflections?:Reflection[];density?:Density|null;selected?:number[];onSelect?:(id:number)=>void;kind?:'structure'|'reciprocal'|'showcase'|'density';showCell?:boolean;ellipsoids?:boolean;showDensity?:boolean;densityOpacity?:number;densityWireframe?:boolean;className?:string;autoRotate?:boolean;controls?:boolean};
const vector=(v:number[])=>new THREE.Vector3(v[0],v[2],-v[1]);
export default function ThreeView(props:Props){
  const host=useRef<HTMLDivElement>(null);const controlsRef=useRef<OrbitControls>();const resetRef=useRef<()=>void>();
  const [error,setError]=useState('');const [loaded,setLoaded]=useState(false);const [rotate,setRotate]=useState(props.autoRotate||false);const [cellVisible,setCellVisible]=useState(props.showCell||false);
  const selectRef=useRef(props.onSelect);selectRef.current=props.onSelect;
  const selectionTargets=useRef<{object:THREE.Object3D;ids:number[];rows?:Reflection[]}[]>([]);
  const densityMaterials=useRef<THREE.MeshStandardMaterial[]>([]);
  const densityPose=useRef<{direction:THREE.Vector3;zoom:number;target:THREE.Vector3;radius:number}>();
  useEffect(()=>{for(const material of densityMaterials.current){material.opacity=props.densityOpacity??.35;material.wireframe=props.densityWireframe??false;material.needsUpdate=true;}},[props.densityOpacity,props.densityWireframe]);
  useEffect(()=>{if(controlsRef.current)controlsRef.current.autoRotate=rotate;},[rotate]);
  useEffect(()=>{
    if(!host.current)return;
    const element=host.current;let disposed=false;setError('');setLoaded(false);
    let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});}catch{setError('当前浏览器未能启用三维显示。');return;}
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.45;
    renderer.domElement.setAttribute('aria-label',props.kind==='reciprocal'?'可旋转倒易空间散点图':props.kind==='density'?'可旋转电子密度等值面':'可旋转晶体结构');renderer.domElement.dataset.testid='three-canvas';element.prepend(renderer.domElement);
    selectionTargets.current=[];densityMaterials.current=[];
    const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(34,element.clientWidth/Math.max(element.clientHeight,1),.01,10000);camera.up.set(0,1,0);
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.1;controls.autoRotate=rotate;controls.autoRotateSpeed=.65;controls.enablePan=true;controlsRef.current=controls;
    scene.add(new THREE.HemisphereLight(0xffffff,0xc3c8c3,2.6));const key=new THREE.DirectionalLight(0xfff5e9,3.2);key.position.set(20,35,28);scene.add(key);const rim=new THREE.DirectionalLight(0xe1eaf0,2.1);rim.position.set(-25,5,-15);scene.add(rim);
    const content=new THREE.Group();scene.add(content);const pickable:THREE.Object3D[]=[];
    function addCell(edges:number[][][]){const positions=edges.flatMap(edge=>edge.flatMap(p=>vector(p).toArray()));const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));content.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:0x9aa49c,transparent:true,opacity:.6})));}
    function addDensity(){if(!props.showDensity||!props.density)return;for(const surface of props.density.surfaces){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(surface.vertices.flatMap(v=>vector(v).toArray()),3));geometry.setIndex(surface.faces.flat());geometry.computeVertexNormals();const material=new THREE.MeshStandardMaterial({color:surface.sign>0?'#477d70':'#b65c43',transparent:true,opacity:props.densityOpacity??.35,wireframe:props.densityWireframe??false,side:THREE.DoubleSide,depthWrite:false});densityMaterials.current.push(material);content.add(new THREE.Mesh(geometry,material));}}
    let fittedRadius=0,lastFitDistance=0;
    const fitDistance=()=>{const vertical=THREE.MathUtils.degToRad(camera.fov/2);const angle=Math.min(vertical,Math.atan(Math.tan(vertical)*camera.aspect));return fittedRadius/Math.sin(angle);};
    function fit(){
      if(disposed)return;
      const box=new THREE.Box3().setFromObject(content);if(box.isEmpty()){setLoaded(true);return;}
      const center=box.getCenter(new THREE.Vector3());content.position.sub(center);
      const sphere=new THREE.Box3().setFromObject(content).getBoundingSphere(new THREE.Sphere());const r=Math.max(sphere.radius,.5);fittedRadius=r;const distance=fitDistance();lastFitDistance=distance;
      camera.near=Math.max(r/1000,.001);camera.far=distance*50;camera.updateProjectionMatrix();
      const reset=()=>{controls.target.set(0,0,0);camera.position.copy(new THREE.Vector3(1.05,.72,1.35).normalize().multiplyScalar(fitDistance()*(props.kind==='showcase'?.94:1.04)));controls.update();};resetRef.current=reset;reset();if(props.kind==='density'&&densityPose.current){const pose=densityPose.current;controls.target.copy(pose.target).multiplyScalar(r/pose.radius);camera.position.copy(pose.direction).multiplyScalar(fitDistance()*pose.zoom).add(controls.target);controls.update();}setLoaded(true);
    }
    if(props.glb){
      new GLTFLoader().load(props.glb,gltf=>{
        if(disposed)return;
        gltf.scene.updateMatrixWorld(true);
        const buckets=new Map<string,{material:THREE.Material;geometries:THREE.BufferGeometry[]}>();
        gltf.scene.traverse(obj=>{if(obj instanceof THREE.Mesh){const material=Array.isArray(obj.material)?obj.material[0]:obj.material;const key=material.uuid+Object.keys(obj.geometry.attributes).sort().join(',')+String(!!obj.geometry.index);if(!buckets.has(key))buckets.set(key,{material,geometries:[]});buckets.get(key)!.geometries.push(obj.geometry.clone().applyMatrix4(obj.matrixWorld));}});
        for(const bucket of buckets.values()){const merged=mergeGeometries(bucket.geometries,false);if(merged){content.add(new THREE.Mesh(merged,bucket.material));bucket.geometries.forEach(g=>g.dispose());}}
        if(cellVisible&&props.cell){const c=props.cell;const corners=[0,1].flatMap(i=>[0,1].flatMap(j=>[0,1].map(k=>[i,j,k].map((_,d)=>i*c[0][d]+j*c[1][d]+k*c[2][d]))));const edges:number[][][]=[];const bits=[0,1].flatMap(i=>[0,1].flatMap(j=>[0,1].map(k=>[i,j,k])));for(let i=0;i<8;i++)for(let j=i+1;j<8;j++)if(bits[i].reduce((s,v,d)=>s+Math.abs(v-bits[j][d]),0)===1)edges.push([corners[i],corners[j]]);addCell(edges);}
        fit();
      },undefined,()=>{if(!disposed)setError('三维模型未能加载，请重新打开页面。');});
    }else if(props.kind==='reciprocal'){
      const rows=props.reflections||[];const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(rows.flatMap(r=>vector(r.q).toArray()),3));
      const selected=new Set(props.selected);const colors=rows.flatMap(r=>new THREE.Color(selected.has(r.id)?'#c95625':r.intensity!==null&&r.intensity<0?'#8394a2':'#576b64').toArray());geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
      const points=new THREE.Points(geometry,new THREE.PointsMaterial({size:5,vertexColors:true,sizeAttenuation:false,transparent:true,opacity:.84}));points.userData.rows=rows;content.add(points);pickable.push(points);
      selectionTargets.current.push({object:points,ids:rows.map(r=>r.id),rows});
      const extent=Math.max(...rows.slice(0,50000).map(r=>Math.hypot(...r.q)),1);const axes=new THREE.AxesHelper(extent*.5);content.add(axes);fit();
    }else if(props.crystal){
      const crystal=props.crystal;const sphereGeometry=new THREE.SphereGeometry(1,20,14);const groups=new Map<string,typeof crystal.atoms>();
      for(const atom of crystal.atoms){if(!groups.has(atom.element))groups.set(atom.element,[]);groups.get(atom.element)!.push(atom);}
      const radii:Record<string,number>={H:.15,C:.32,O:.34,N:.34,Zn:.56,Cu:.52,Fe:.48,Zr:.61};
      for(const atoms of groups.values()){
        const mesh=new THREE.InstancedMesh(sphereGeometry,new THREE.MeshStandardMaterial({color:atoms[0].color,roughness:.35,metalness:.04}),atoms.length);
        atoms.forEach((atom,i)=>{const position=vector(atom.position),radius=(radii[atom.element]||.4)*(props.kind==='density'?.6:1);let quaternion=new THREE.Quaternion(),scale=new THREE.Vector3(radius,radius,radius);if(props.ellipsoids&&atom.ellipsoid){const a=atom.ellipsoid.axes;const basis=new THREE.Matrix4().makeBasis(vector([a[0][0],a[1][0],a[2][0]]),vector([a[0][1],a[1][1],a[2][1]]),vector([a[0][2],a[1][2],a[2][2]]));quaternion.setFromRotationMatrix(basis);scale=new THREE.Vector3(...atom.ellipsoid.radii as [number,number,number]);}const matrix=new THREE.Matrix4().compose(position,quaternion,scale);mesh.setMatrixAt(i,matrix);if(props.selected?.includes(atom.id))mesh.setColorAt(i,new THREE.Color('#e18a4e'));});
        mesh.userData.atomIds=atoms.map(a=>a.id);mesh.instanceMatrix.needsUpdate=true;content.add(mesh);pickable.push(mesh);
        selectionTargets.current.push({object:mesh,ids:atoms.map(a=>a.id)});
      }
      const rods=new THREE.InstancedMesh(new THREE.CylinderGeometry(.065,.065,1,8),new THREE.MeshStandardMaterial({color:'#9ba39e',roughness:.5}),crystal.bonds.length);const up=new THREE.Vector3(0,1,0);
      crystal.bonds.forEach(([i,j],index)=>{const a=vector(crystal.atoms[i].position),b=vector(crystal.atoms[j].position),d=b.clone().sub(a);rods.setMatrixAt(index,new THREE.Matrix4().compose(a.add(b).multiplyScalar(.5),new THREE.Quaternion().setFromUnitVectors(up,d.clone().normalize()),new THREE.Vector3(1,d.length(),1)));});content.add(rods);
      if(cellVisible)addCell(crystal.cellEdges);
      addDensity();
      fit();
    }else{addDensity();if(cellVisible&&props.density?.cellEdges)addCell(props.density.cellEdges);fit();}
    const raycaster=new THREE.Raycaster();let down={x:0,y:0};
    const pointerDown=(e:PointerEvent)=>{down={x:e.clientX,y:e.clientY};};
    const pointerUp=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down.x,e.clientY-down.y)>5)return;const box=renderer.domElement.getBoundingClientRect();const p=new THREE.Vector2((e.clientX-box.left)/box.width*2-1,-(e.clientY-box.top)/box.height*2+1);raycaster.setFromCamera(p,camera);raycaster.params.Points={threshold:new THREE.Box3().setFromObject(content).getSize(new THREE.Vector3()).length()*.012};const hit=raycaster.intersectObjects(pickable,false)[0];if(hit){if(hit.object instanceof THREE.InstancedMesh&&hit.instanceId!==undefined)selectRef.current?.(hit.object.userData.atomIds[hit.instanceId]);else if(hit.index!==undefined)selectRef.current?.(hit.object.userData.rows[hit.index].id);}};
    renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp);
    const resize=()=>{const w=element.clientWidth,h=element.clientHeight;renderer.setSize(w,h);camera.aspect=Math.max(w,1)/Math.max(h,1);if(fittedRadius){const next=fitDistance();camera.position.sub(controls.target).multiplyScalar(next/lastFitDistance).add(controls.target);lastFitDistance=next;camera.far=next*50;}camera.updateProjectionMatrix();};const observer=new ResizeObserver(resize);observer.observe(element);resize();
    renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
    return()=>{if(props.kind==='density'&&lastFitDistance&&fittedRadius){const offset=camera.position.clone().sub(controls.target);densityPose.current={direction:offset.clone().normalize(),zoom:offset.length()/lastFitDistance,target:controls.target.clone(),radius:fittedRadius};}disposed=true;observer.disconnect();renderer.setAnimationLoop(null);controls.dispose();scene.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Points||o instanceof THREE.LineSegments){o.geometry.dispose();(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose());}});renderer.dispose();renderer.domElement.remove();controlsRef.current=undefined;};
  },[props.glb,props.crystal,props.reflections,props.density,props.showDensity,props.ellipsoids,cellVisible]);
  useEffect(()=>{
    const selected=new Set(props.selected||[]);
    for(const target of selectionTargets.current){
      if(target.object instanceof THREE.InstancedMesh){target.ids.forEach((id,i)=>target.object instanceof THREE.InstancedMesh&&target.object.setColorAt(i,new THREE.Color(selected.has(id)?'#f0b284':'#ffffff')));if(target.object.instanceColor)target.object.instanceColor.needsUpdate=true;}
      else if(target.object instanceof THREE.Points){const colors=target.object.geometry.getAttribute('color') as THREE.BufferAttribute;target.ids.forEach((id,i)=>{const c=new THREE.Color(selected.has(id)?'#d46932':target.rows?.[i].intensity!==null&&(target.rows?.[i].intensity||0)<0?'#8394a2':'#576b64');colors.setXYZ(i,c.r,c.g,c.b);});colors.needsUpdate=true;}
    }
  },[props.selected]);
  return <div className={`three-view ${props.className||''}`} ref={host} data-loaded={loaded} data-surface-count={props.showDensity?props.density?.surfaces.length||0:0}>
    {error?(props.poster?<><img className="model-poster" alt="晶体结构预览" src={props.poster}/><p className="canvas-error">{error}</p></>:<p className="canvas-error">{error}</p>):!loaded&&<div className="canvas-loading">正在载入三维结构…</div>}
    {props.controls!==false&&<div className="canvas-tools"><button className={`icon-button ${cellVisible?'selected':''}`} title="显示或隐藏晶胞" aria-label="显示晶胞" aria-pressed={cellVisible} onClick={()=>setCellVisible(v=>!v)}><Box size={21}/></button><button className={`icon-button ${rotate?'selected':''}`} title="自动旋转" aria-label="自动旋转" aria-pressed={rotate} onClick={()=>setRotate(v=>!v)}><Rotate3D size={22}/></button><button className="icon-button" title="重置视角" aria-label="重置视角" onClick={()=>resetRef.current?.()}><RotateCcw size={21}/></button></div>}
  </div>;
}
