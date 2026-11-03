#!/usr/bin/env python3
"""Continuously audit MAS-Gate phase concurrency, throughput, exclusivity and GPU use."""
from __future__ import annotations
import json, os, re, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

INTERVAL=max(1.0,float(os.getenv('MAS_GATE_PHASE_MONITOR_INTERVAL','2')))
CAP=int(os.getenv('MAS_GATE_PHASE_CONCURRENCY','128'))
MEMORY_TARGET=float(os.getenv('MAS_GATE_ACTIVE_GPU_MEMORY_MIN_FRACTION','0.90'))
CLIENT_CAPS={
 'orchestra':int(os.getenv('MAS_GATE_ORCHESTRA_AGGREGATE_CONCURRENCY',str(CAP))),
 'sub_agent':int(os.getenv('MAS_GATE_SUB_AGENT_AGGREGATE_CONCURRENCY',str(CAP))),
 'judge':int(os.getenv('MAS_GATE_JUDGE_AGGREGATE_CONCURRENCY',str(CAP))),
 'qwen_j':int(os.getenv('MAS_GATE_QWEN_J_AGGREGATE_CONCURRENCY',str(CAP))),
 'bge':int(os.getenv('MAS_GATE_BGE_AGGREGATE_CONCURRENCY',str(CAP))),
}
SERVER_CAPS={
 'orchestra':int(os.getenv('MAS_GATE_ORCHESTRA_SERVER_MAX_NUM_SEQS',str(CAP))),
 'sub_agent':int(os.getenv('MAS_GATE_SUB_AGENT_SERVER_MAX_NUM_SEQS',str(CAP))),
 'judge':int(os.getenv('MAS_GATE_JUDGE_SERVER_MAX_NUM_SEQS',str(CAP))),
 'qwen_j':int(os.getenv('MAS_GATE_QWEN_J_SERVER_MAX_NUM_SEQS',str(CAP))),
 'bge':int(os.getenv('MAS_GATE_BGE_SERVER_MAX_NUM_SEQS',str(CAP))),
}
STATE=Path(os.getenv('MAS_GATE_STATE_FILE','/tmp/mas_gate_rl_current_run.env'))
PHASE=Path(os.getenv('MAS_GATE_REWARD_PHASE_FILE','/tmp/mas_gate_reward_phase.json'))
PROM=Path(os.getenv('ORCHESTRA_PROMETHEUS_CONFIG_FILE','/tmp/mas_gate_orchestra_prometheus.yml'))
URLS={
 'sub_agent':os.getenv('MAS_GATE_SUB_AGENT_METRICS_URL','http://127.0.0.1:8000').rstrip('/'),
 'judge':os.getenv('MAS_GATE_JUDGE_METRICS_URL','http://127.0.0.1:8003').rstrip('/'),
 'bge':os.getenv('MAS_GATE_BGE_METRICS_URL','http://127.0.0.1:8002').rstrip('/'),
}
GPUS={int(x) for x in os.getenv('ALL_GPUS','0,1,2,3,4,5,6,7').split(',')}
COUNTERS=('request_success_total','prompt_tokens_total','generation_tokens_total')

def state_value(key:str)->str:
    try:
        for line in STATE.read_text().splitlines():
            if line.startswith(key+'='): return line.split('=',1)[1]
    except Exception: pass
    return ''
root=Path(os.getenv('MAS_GATE_PHASE_MONITOR_ROOT') or state_value('ARTIFACT_ROOT') or '/tmp')
out=Path(os.getenv('MAS_GATE_PHASE_MONITOR_LOG') or root/'monitor'/'phase_capacity.jsonl')
summary=Path(os.getenv('MAS_GATE_PHASE_MONITOR_SUMMARY') or root/'monitor'/'phase_capacity_latest.json')
out.parent.mkdir(parents=True,exist_ok=True)

def fetch(url:str,timeout:float=1.5)->str:
    try:
        with urllib.request.urlopen(url,timeout=timeout) as r: return r.read().decode('utf-8','replace')
    except Exception: return ''
def vals(text:str,name:str):
    ans=[]
    for raw in re.findall(rf'^{re.escape(name)}(?:\{{[^\n]*\}})?\s+([0-9.eE+-]+)$',text,re.M):
        try: ans.append(float(raw))
        except ValueError: pass
    return ans
def metric_sum(text:str,*names:str)->float:
    for name in names:
        found=vals(text,name)
        if found: return sum(found)
    return 0.0
def service(url:str):
    text=fetch(url+'/metrics'); state=fetch(url+'/is_sleeping')
    sm=re.search(r'true|false',state,re.I)
    kv=vals(text,'vllm:kv_cache_usage_perc') or vals(text,'vllm:gpu_cache_usage_perc')
    return {'available':bool(text or state),
            'sleeping':(sm.group(0).lower()=='true') if sm else None,
            'running':sum(vals(text,'vllm:num_requests_running')),
            'waiting':sum(vals(text,'vllm:num_requests_waiting')),
            'kv_peak':max(kv) if kv else None,
            'request_success_total':metric_sum(text,'vllm:request_success_total','vllm:request_success_created'),
            'prompt_tokens_total':metric_sum(text,'vllm:prompt_tokens_total','vllm:prompt_tokens_created'),
            'generation_tokens_total':metric_sum(text,'vllm:generation_tokens_total','vllm:generation_tokens_created')}
def targets():
    try: text=PROM.read_text()
    except Exception: return []
    return list(dict.fromkeys(re.findall(r"^\s*-\s*['\"]?([0-9A-Fa-f:.]+:\d+)['\"]?\s*$",text,re.M)))
def combine(rows:list[dict]):
    kv=[r['kv_peak'] for r in rows if r['kv_peak'] is not None]
    return {'targets':len(rows),'available':any(r['available'] for r in rows),
            'sleeping':None,'running':sum(r['running'] for r in rows),
            'waiting':sum(r['waiting'] for r in rows),'kv_peak':max(kv) if kv else None,
            **{k:sum(r[k] for r in rows) for k in COUNTERS}}
def orchestra():
    ts=targets()
    with ThreadPoolExecutor(max_workers=max(1,min(16,len(ts)))) as ex:
        rows=list(ex.map(lambda t:service('http://'+t),ts)) if ts else []
    return combine(rows)
def gpu():
    try:
        cp=subprocess.run(['gpustat','--json'],capture_output=True,text=True,timeout=8,check=True)
        rows=[g for g in json.loads(cp.stdout).get('gpus',[]) if int(g.get('index',-1)) in GPUS]
        util=[float(g.get('utilization.gpu') or 0) for g in rows]
        mem=[float(g.get('memory.used') or 0) for g in rows]
        total=[float(g.get('memory.total') or 0) for g in rows]
        total_mib=sum(total)
        return {'avg_util':sum(util)/len(util) if util else 0,'max_util':max(util,default=0),
                'mem_used_mib':sum(mem),'mem_total_mib':total_mib,
                'mem_fraction':(sum(mem)/total_mib) if total_mib else 0,
                'mem_per_gpu_mib':mem,'mem_total_per_gpu_mib':total}
    except Exception: return {'avg_util':0,'max_util':0,'mem_used_mib':0,'mem_total_mib':0,
                              'mem_fraction':0,'mem_per_gpu_mib':[],'mem_total_per_gpu_mib':[]}
def phase():
    try: return str(json.loads(PHASE.read_text()).get('phase','full'))
    except Exception: return 'full'
def deltas(name:str,row:dict,previous:dict[str,dict])->dict:
    before=previous.get(name,{})
    ans={}
    for key in COUNTERS:
        cur=float(row.get(key,0)); old=float(before.get(key,cur))
        ans[key.replace('_total','_delta')]=max(0.0,cur-old) if cur>=old else max(0.0,cur)
    previous[name]={key:float(row.get(key,0)) for key in COUNTERS}
    return ans

def new_peak():
    return {'samples':0,'peak_running':0,'peak_waiting':0,'peak_kv':None,'peak_gpu_avg':0,
            'peak_gpu_mem_mib':0,'peak_gpu_mem_fraction':0,'request_success_delta':0,'prompt_tokens_delta':0,
            'generation_tokens_delta':0,'peak_requests_per_interval':0,
            'peak_prompt_tokens_per_interval':0,'peak_generation_tokens_per_interval':0,'violations':0}

def load_existing():
    try:
        payload=json.loads(summary.read_text())
        return payload.get('phase_peaks',{}),payload.get('recent_violations',[])
    except Exception:
        return {},[]

def phase_service_name(phase_name):
    return {'full':'orchestra','sub_agent':'sub_agent','bcp_llm_judge':'judge',
            'qwen_j':'qwen_j','bge_impact':'bge'}.get(phase_name,'orchestra')

def capacity_status(items):
    result={}
    for phase_name,st in items.items():
        service_name=phase_service_name(phase_name)
        client_cap=CLIENT_CAPS[service_name]
        server_cap=SERVER_CAPS[service_name]
        request_activity=st.get('request_success_delta',0)>0
        compute_saturated=st.get('peak_gpu_avg',0)>=90
        observed_queue=st.get('peak_running',0)+st.get('peak_waiting',0)
        result[phase_name]={'configured_cap':CAP,
                  'service':service_name,
                  'client_configured_concurrency':client_cap,
                  'server_configured_max_num_seqs':server_cap,
                  'configured_aggregate_concurrency_ok':client_cap==CAP and server_cap==CAP,
                  'peak_running':st.get('peak_running',0),
                  'peak_running_plus_waiting':observed_queue,
                  'request_activity':request_activity,
                  'peak_requests_per_interval':st.get('peak_requests_per_interval',0),
                  'peak_prompt_tokens_per_interval':st.get('peak_prompt_tokens_per_interval',0),
                  'peak_gpu_avg':st.get('peak_gpu_avg',0),
                  'peak_gpu_mem_fraction':st.get('peak_gpu_mem_fraction',0),
                  'near_full_memory_observed':st.get('peak_gpu_mem_fraction',0)>=MEMORY_TARGET,
                  'compute_saturation_observed':compute_saturated,
                  # Prompt-logprob and embedding requests can finish between
                  # one-second scrapes, so running=0 is not evidence of zero
                  # concurrency.  Configured 128 + request deltas + saturated
                  # compute is the appropriate capacity evidence there.
                  'short_request_capacity_evidence':(
                      client_cap==CAP and server_cap==CAP and request_activity and compute_saturated
                  )}
    return result

peaks,violations=load_existing(); previous={}
while True:
    now=time.time(); ph=phase()
    with ThreadPoolExecutor(max_workers=3) as ex:
        fs={k:ex.submit(service,u) for k,u in URLS.items()}
        services={k:f.result() for k,f in fs.items()}
    orch=orchestra(); g=gpu()
    active={'full':'orchestra','sub_agent':'sub_agent','bcp_llm_judge':'judge','qwen_j':'sub_agent','bge_impact':'bge'}.get(ph,'orchestra')
    rows={'orchestra':orch,**services}
    for name,row in rows.items(): row.update(deltas(name,row,previous))
    ar=rows[active]; running=ar['running']; cur=[]
    if running>CAP: cur.append(f'{active}.running={running}>{CAP}')
    for name,row in services.items():
        # A single fresh Qwen endpoint can intentionally serve both sub-agent
        # and BCP judge roles.  Treat services with the same metrics URL as one
        # physical service; otherwise the alias is falsely reported awake while
        # its peer role is active.
        should_awake=(name==active) or (
            active in URLS and URLS.get(name)==URLS.get(active)
        )
        if not should_awake and row['sleeping'] is False: cur.append(f'{name}.awake_while_inactive')
    rec={'time':datetime.now().astimezone().isoformat(timespec='seconds'),'unix':now,'phase':ph,'active':active,
         'cap':CAP,'orchestra':orch,'services':services,'gpu':g,'violations':cur}
    # Checkpoint/NFS cleanup may briefly recreate the monitor directory.  Rebuild
    # it before every append instead of letting the long-running capacity audit die.
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    st=peaks.setdefault(ph,new_peak())
    st['samples']+=1; st['peak_running']=max(st['peak_running'],ar['running']); st['peak_waiting']=max(st['peak_waiting'],ar['waiting'])
    if ar['kv_peak'] is not None: st['peak_kv']=max(st['peak_kv'] or 0,ar['kv_peak'])
    st['peak_gpu_avg']=max(st['peak_gpu_avg'],g['avg_util']); st['peak_gpu_mem_mib']=max(st['peak_gpu_mem_mib'],g['mem_used_mib'])
    st['peak_gpu_mem_fraction']=max(st.get('peak_gpu_mem_fraction',0),g.get('mem_fraction',0))
    for key in ('request_success','prompt_tokens','generation_tokens'):
        delta=ar[key+'_delta']; st[key+'_delta']+=delta
        peak_key={'request_success':'peak_requests_per_interval','prompt_tokens':'peak_prompt_tokens_per_interval','generation_tokens':'peak_generation_tokens_per_interval'}[key]
        st[peak_key]=max(st[peak_key],delta)
    st['violations']+=len(cur)
    if cur: violations.append({'time':rec['time'],'phase':ph,'items':cur})
    # Backfill the normalized fraction for peaks written by pre-upgrade monitors.
    if g.get('mem_total_mib'):
        for old_st in peaks.values():
            if old_st.get('peak_gpu_mem_mib'):
                old_st['peak_gpu_mem_fraction']=max(old_st.get('peak_gpu_mem_fraction',0),old_st['peak_gpu_mem_mib']/g['mem_total_mib'])
    payload={'updated':rec['time'],'cap':CAP,'client_configured_concurrency':CLIENT_CAPS,
             'server_configured_max_num_seqs':SERVER_CAPS,
             'memory_target_fraction':MEMORY_TARGET,'interval_seconds':INTERVAL,'urls':URLS,'gpus':sorted(GPUS),
             'current':rec,'phase_peaks':peaks,'phase_capacity_status':capacity_status(peaks),
             'recent_violations':violations[-30:]}
    # Publish atomically so readers never observe a truncated JSON document and
    # tolerate the output directory being replaced while checkpoints are saved.
    summary.parent.mkdir(parents=True,exist_ok=True)
    summary_tmp=summary.with_name(f'.{summary.name}.tmp.{os.getpid()}')
    summary_tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(summary_tmp,summary)
    time.sleep(max(0,INTERVAL-(time.time()-now)))
