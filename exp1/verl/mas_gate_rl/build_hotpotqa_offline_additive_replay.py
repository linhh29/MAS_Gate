#!/usr/bin/env python3
"""Build fixed-rollout HotpotQA replay groups for I=U+(1+K) ablation.

This is deliberately additive-only and never edits the source checkpoint.
Each output row is one GRPO prompt group containing 16 historical responses.
"""
from __future__ import annotations
import argparse, json, math, hashlib
from pathlib import Path
from collections import defaultdict


def read_jsonl(p):
    with open(p, encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def finite(x):
    try:
        x=float(x); return math.isfinite(x)
    except Exception: return False

def additive_mean(q):
    nodes=q.get('nodes') if isinstance(q,dict) else None
    vals=[]
    if isinstance(nodes,list):
        for n in nodes:
            if finite(n.get('U_i')) and finite(n.get('K')):
                vals.append(float(n['U_i']) + (1.0 + float(n['K'])))
    if vals: return sum(vals)/len(vals)
    # Some traces retain only the aggregate; do not silently use the old impact.
    return None

def z(vals):
    m=sum(vals)/len(vals)
    var=sum((x-m)**2 for x in vals)/len(vals)
    s=math.sqrt(var)
    return [(x-m)/s if s>1e-12 else 0.0 for x in vals],m,s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--trace-root', required=True)
    ap.add_argument('--group-size', type=int, default=16)
    ap.add_argument('--max-steps', type=int, default=0)
    args=ap.parse_args()
    src=Path(args.source); out=Path(args.output); tr=Path(args.trace_root)
    fusion=defaultdict(list)
    for r in read_jsonl(src):
        fusion[int(r['step'])].append(r)
    # Prefer exact solution hash, then task/index. The result filename convention is
    # <index>__<task_id>__<solution_sha1>.json.
    qr_by_idx=defaultdict(list)
    for p in sorted((tr/'reward_results').glob('*.json')):
        try:
            idx=int(p.name.split('__',1)[0]); qr_by_idx[idx].append(p)
        except Exception: continue
    trace_by_idx=defaultdict(list)
    for p in sorted((tr/'sample_trace_json').glob('*.json')):
        try:
            idx=int(p.name.split('__',1)[0]); trace_by_idx[idx].append(p)
        except Exception: continue
    groups=[]; stats=defaultdict(int); missing=[]
    for step in sorted(fusion):
        if args.max_steps and step>args.max_steps: continue
        rows=sorted(fusion[step], key=lambda x:int(x.get('row',0)))
        if len(rows)%args.group_size: raise RuntimeError(f'step {step}: {len(rows)} rows not divisible by {args.group_size}')
        for gi in range(0,len(rows),args.group_size):
            rr=rows[gi:gi+args.group_size]; items=[]; dense=[]; corr=[]
            for r in rr:
                uid=str(r.get('uid','')); chosen=None
                # reward_fusion row order is the historical rollout order: 16
                # responses for each dataset index.  The source step is a
                # contiguous 256-prompt window.
                idx=(step-1)*256 + int(r.get('row',0))//args.group_size
                rp=qr_by_idx.get(idx,[])
                if rp:
                    # Keep a deterministic response order by filename; this is
                    # the same index/rollout ordering used by the trace writer.
                    p=rp[int(r.get('row',0)) % args.group_size % len(rp)]
                    d=json.loads(p.read_text(encoding='utf-8')); q=d.get('result',{}).get('quant_result')
                    if isinstance(q,dict) and additive_mean(q) is not None: chosen=(d,q)
                if chosen is None:
                    cand=rp
                    for p in cand:
                        try:
                            d=json.loads(p.read_text(encoding='utf-8')); q=d.get('result',{}).get('quant_result')
                            if isinstance(q,dict) and additive_mean(q) is not None: chosen=(d,q); break
                        except Exception: pass
                if chosen is None:
                    missing.append((step,int(r.get('row',-1)),uid)); continue
                d,q=chosen; dm=additive_mean(q)
                idx=int(d.get('index', idx))
                tp=trace_by_idx.get(idx,[])
                tc=[]
                if tp:
                    sha=str(d.get('solution_sha1',''))
                    ordered=[candidate for candidate in tp if sha and sha in candidate.name] + [candidate for candidate in tp if not (sha and sha in candidate.name)]
                    for candidate in ordered:
                        try:
                            t0=json.loads(candidate.read_text(encoding='utf-8'))
                            if t0.get('success') and t0.get('orchestra_rollout_xml'):
                                tc.append(t0)
                                if sha and sha in candidate.name: break
                        except Exception: pass
                if not tc:
                    missing.append((step,int(r.get('row',-1)),uid)); continue
                t=tc[0]
                item={'index':idx,'task_id':str(t.get('task_id',d.get('task_id',''))),'question':t.get('question',''),
                      'ground_truth':t.get('ground_truth',''),'response':t['orchestra_rollout_xml'],
                      'additive_dense_reward':dm,'correctness_reward':float(r.get('raw_correctness_reward',t.get('judge_score',0.0))),
                      'old_uid':uid,'source_step':step,'source_row':int(r.get('row',-1))}
                items.append(item); dense.append(dm); corr.append(item['correctness_reward'])
            if len(items)!=args.group_size:
                stats['incomplete_groups']+=1; continue
            zd,dm,ds=z(dense); zc,cm,cs=z(corr)
            for item,a,b in zip(items,zd,zc): item['replay_reward']=a+b
            groups.append({'group_id':f'step{step:04d}_group{gi//args.group_size:04d}','source_step':step,'items':items,
                           'group_size':args.group_size,'dense_mean':dm,'dense_std':ds,'correctness_mean':cm,'correctness_std':cs})
            stats['groups']+=1; stats['items']+=len(items)
    if missing:
        print(f'WARNING missing_or_unmatched={len(missing)}', flush=True)
        if len(missing)>10: raise RuntimeError('Too many unmatched replay rows; refusing unsafe training input')
    out.mkdir(parents=True,exist_ok=True)
    with open(out/'groups.jsonl','w',encoding='utf-8') as f:
        for g in groups: f.write(json.dumps(g,ensure_ascii=False)+'\n')
    manifest={'formula':'I=U+(1+K)','group_size':args.group_size,'source':str(src),'trace_root':str(tr),**stats,
              'sha256':hashlib.sha256((out/'groups.jsonl').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
