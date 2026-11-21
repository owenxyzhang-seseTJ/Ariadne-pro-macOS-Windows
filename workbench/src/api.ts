import {useCallback,useEffect,useState} from 'react';
export async function api<T>(url:string,body?:unknown,method='POST'):Promise<T>{
  const options:RequestInit=body===undefined?{}:{method,headers:body instanceof FormData?{}:{'Content-Type':'application/json'},body:body instanceof FormData?body:JSON.stringify(body)};
  const response=await fetch(url,options);let data:unknown;
  try{data=await response.json();}catch{throw new Error('本地服务返回了无法读取的响应。');}
  if(!response.ok)throw new Error((data as {error?:string}).error||`请求失败 (${response.status})`);
  return data as T;
}
export function useData<T>(url:string|null,version=0){
  const [data,setData]=useState<T|null>(null);const [error,setError]=useState('');const [loading,setLoading]=useState(false);
  const reload=useCallback(async()=>{if(!url)return;setLoading(true);try{setData(await api<T>(url));setError('');}catch(e){setError((e as Error).message);}finally{setLoading(false);}},[url]);
  useEffect(()=>{let current=true;if(!url){setData(null);return;}setLoading(true);setData(null);api<T>(url).then(d=>{if(current){setData(d);setError('');}}).catch(e=>{if(current)setError(e.message);}).finally(()=>{if(current)setLoading(false);});return()=>{current=false;};},[url,version]);
  return {data,error,loading,reload,setData};
}
export const bytes=(value:number)=>value>1024*1024?`${(value/1024/1024).toFixed(1)} MB`:`${Math.round(value/1024)} KB`;
export const number=(value:unknown,digits=3)=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('en-US',{maximumFractionDigits:digits}):'—';
