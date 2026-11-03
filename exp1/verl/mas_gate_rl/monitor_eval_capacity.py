#!/usr/bin/env python3
"""Continuously audit a phase-exclusive MAS-Gate evaluation lane."""
from __future__ import annotations
import json, os, re, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

INTERVAL=max(1.0,float(os.getenv('MAS_GATE_EVAL_MONITOR_INTERVAL','2')))
CAP=int(os.getenv('MAS_GATE_EVAL_CONCURRENCY','128'))
MEMORY_TARGET=float(os.getenv('MAS_GATE_ACTIVE_GPU_MEMORY_MIN_FRACTION','0.90'))
ROLE_URLS={
 'orchestra':os.getenv('MAS_GATE_ORCHESTRA_METRICS_URL','http://127.0.0.1:8201').rstrip('/'),
 'sub_agent':os.getenv('MAS_GATE_SUB_AGENT_METRICS_URL','http://127.0.0.1:8200').rstrip('/'),
 'judge':os.getenv('MAS_GATE_JUDGE_METRICS_URL','http://127.0.0.1:8203').rstrip('/'),
}
# A semantic evaluation deliberately aliases judge and sub-agent to one
# persistent Qwen service. Audit physical endpoints, not logical roles, so the
# shared endpoint is never misreported as two simultaneously-awake models.
URL_ROLES={}
for _role,_url in ROLE_URLS.items(): URL_ROLES.setdefault(_url,[]).append(_role)
URLS={'+'.join(_roles):_url for _url,_roles in URL_ROLES.items()}
GPUS={int(x) for x in os.getenv('ALL_GPUS',os.getenv('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7')).split(',')}
ROOT=Path(os.getenv('MAS_GATE_EVAL_MONITOR_ROOT','/tmp/mas_gate_eval_monitor'))
OUT=Path(os.getenv('MAS_GATE_EVAL_MONITOR_LOG') or ROOT/'capacity.jsonl')
SUMMARY=Path(os.getenv('MAS_GATE_EVAL_MONITOR_SUMMARY') or ROOT/'capacity_latest.json')
COUNTERS=('request_success_total','prompt_tokens_total','generation_tokens_total')
OUT.parent.mkdir(parents=True,exist_ok=True)

def fetch(url:str,timeout:float=1.5)->str:
    try:
        with urllib.request.urlopen(url,timeout=timeout) as r:return r.read().decode('utf-8','replace')
    except Exception:return ''
def vals(text:str,name:str):
    out=[]
    for raw in re.findall(rf'^{re.escape(name)}(?:\{{[^\n]*\}})?\s+([0-9.eE+-]+)$',text,re.M):
        try:out.append(float(raw))
        except ValueError:pass
    return out
def metric_sum(text:str,*names:str)->float:
    for name in names:
        found=vals(text,name)
        if found:return sum(found)
    return 0.0
def service(url:str):
    text=fetch(url+'/metrics'); state=fetch(url+'/is_sleeping')
    sm=re.search(r'true|false',state,re.I)
    kv=vals(text,'vllm:kv_cache_usage_perc') or vals(text,'vllm:gpu_cache_usage_perc')
    return {'available':bool(text or state),'sleeping':(sm.group(0).lower()=='true') if sm else None,
            'running':sum(vals(text,'vllm:num_requests_running')),
            'waiting':sum(vals(text,'vllm:num_requests_waiting')),'kv_peak':max(kv) if kv else None,
            'request_success_total':metric_sum(text,'vllm:request_success_total','vllm:request_success_created'),
            'prompt_tokens_total':metric_sum(text,'vllm:prompt_tokens_total','vllm:prompt_tokens_created'),
            'generation_tokens_total':metric_sum(text,'vllm:generation_tokens_total','vllm:generation_tokens_created')}
def gpu():
    try:
        cp=subprocess.run(['gpustat','--json'],capture_output=True,text=True,timeout=8,check=True)
        rows=[g for g in json.loads(cp.stdout).get('gpus',[]) if int(g.get('index',-1)) in GPUS]
        util=[float(g.get('utilization.gpu') or 0) for g in rows]; mem=[float(g.get('memory.used') or 0) for g in rows]
        total=[float(g.get('memory.total') or 0) for g in rows]; total_mib=sum(total)
        return {'avg_util':sum(util)/len(util) if util else 0,'max_util':max(util,default=0),
                'mem_used_mib':sum(mem),'mem_total_mib':total_mib,
                'mem_fraction':(sum(mem)/total_mib) if total_mib else 0,
                'mem_per_gpu_mib':mem,'mem_total_per_gpu_mib':total}
    except Exception:return {'avg_util':0,'max_util':0,'mem_used_mib':0,'mem_total_mib':0,
                             'mem_fraction':0,'mem_per_gpu_mib':[],'mem_total_per_gpu_mib':[]}
def add_deltas(name,row,previous):
    before=previous.get(name,{})
    for key in COUNTERS:
        cur=float(row.get(key,0)); old=float(before.get(key,cur))
        row[key.replace('_total','_delta')]=max(0.0,cur-old) if cur>=old else max(0.0,cur)
    previous[name]={key:float(row.get(key,0)) for key in COUNTERS}
def new_peak():
    return {'samples':0,'peak_running':0,'peak_waiting':0,'peak_kv':None,'peak_gpu_avg':0,
            'peak_gpu_mem_mib':0,'peak_gpu_mem_fraction':0,'request_success_delta':0,'prompt_tokens_delta':0,
            'generation_tokens_delta':0,'peak_requests_per_interval':0,
            'peak_prompt_tokens_per_interval':0,'peak_generation_tokens_per_interval':0,'violations':0}

def load_existing():
    try:
        payload=json.loads(SUMMARY.read_text())
        return payload.get('phase_peaks',{}),payload.get('recent_violations',[])
    except Exception:
        return {},[]

def capacity_status(items):
    return {name:{'configured_cap':CAP,
                  'peak_running':st.get('peak_running',0),
                  'peak_running_plus_waiting':st.get('peak_running',0)+st.get('peak_waiting',0),
                  'request_activity':st.get('request_success_delta',0)>0,
                  'peak_gpu_avg':st.get('peak_gpu_avg',0),
                  'peak_gpu_mem_fraction':st.get('peak_gpu_mem_fraction',0),
                  'near_full_memory_observed':st.get('peak_gpu_mem_fraction',0)>=MEMORY_TARGET,
                  'compute_saturation_observed':st.get('peak_gpu_avg',0)>=90}
            for name,st in items.items()}

peaks,violations=load_existing(); previous={}
while True:
    now=time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        fs={k:ex.submit(service,u) for k,u in URLS.items()}; services={k:f.result() for k,f in fs.items()}
    for name,row in services.items():add_deltas(name,row,previous)
    awake=[k for k,v in services.items() if v['sleeping'] is False]
    active=awake[0] if len(awake)==1 else ('initializing' if not awake else 'multiple')
    g=gpu(); cur=[]
    if len(awake)>1:cur.append('multiple_awake='+','.join(awake))
    for name,row in services.items():
        if row['running']>CAP:cur.append(f'{name}.running={row["running"]}>{CAP}')
        if row['sleeping'] is True and (row['running'] or row['waiting']):cur.append(f'{name}.sleeping_with_requests')
    rec={'time':datetime.now().astimezone().isoformat(timespec='seconds'),'unix':now,'phase':active,
         'cap':CAP,'services':services,'gpu':g,'violations':cur}
    with OUT.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    if active in services:
        ar=services[active]; st=peaks.setdefault(active,new_peak())
        st['samples']+=1; st['peak_running']=max(st['peak_running'],ar['running']); st['peak_waiting']=max(st['peak_waiting'],ar['waiting'])
        if ar['kv_peak'] is not None:st['peak_kv']=max(st['peak_kv'] or 0,ar['kv_peak'])
        st['peak_gpu_avg']=max(st['peak_gpu_avg'],g['avg_util']); st['peak_gpu_mem_mib']=max(st['peak_gpu_mem_mib'],g['mem_used_mib'])
        st['peak_gpu_mem_fraction']=max(st.get('peak_gpu_mem_fraction',0),g.get('mem_fraction',0))
        for key in ('request_success','prompt_tokens','generation_tokens'):
            delta=ar[key+'_delta']; st[key+'_delta']+=delta
            peak_key={'request_success':'peak_requests_per_interval','prompt_tokens':'peak_prompt_tokens_per_interval','generation_tokens':'peak_generation_tokens_per_interval'}[key]
            st[peak_key]=max(st[peak_key],delta)
        st['violations']+=len(cur)
    if cur:violations.append({'time':rec['time'],'phase':active,'items':cur})
    # Backfill the normalized fraction for peaks written by pre-upgrade monitors.
    if g.get('mem_total_mib'):
        for old_st in peaks.values():
            if old_st.get('peak_gpu_mem_mib'):
                old_st['peak_gpu_mem_fraction']=max(old_st.get('peak_gpu_mem_fraction',0),old_st['peak_gpu_mem_mib']/g['mem_total_mib'])
    SUMMARY.write_text(json.dumps({'updated':rec['time'],'cap':CAP,'memory_target_fraction':MEMORY_TARGET,'interval_seconds':INTERVAL,'urls':URLS,'role_urls':ROLE_URLS,
        'gpus':sorted(GPUS),'current':rec,'phase_peaks':peaks,'phase_capacity_status':capacity_status(peaks),
        'recent_violations':violations[-30:]},ensure_ascii=False,indent=2)+'\n')
    time.sleep(max(0,INTERVAL-(time.time()-now)))
