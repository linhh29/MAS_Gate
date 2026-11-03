#!/usr/bin/env python3
"""Isolated fixed-rollout replay trainer for the HotpotQA additive ablation."""
from __future__ import annotations
import argparse, asyncio, json, os
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset
from verl import DataProto
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.experimental.separation.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from mas_gate_rl.xml_policy import ORCHESTRA_XML_SYSTEM_PROMPT, build_orchestra_user_prompt

class ReplayDataset(Dataset):
    def __init__(self, path):
        self.rows=[json.loads(x) for x in open(path,encoding='utf-8') if x.strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def collate_groups(xs): return {'groups': xs}

def _pad_left(xs, value):
    n=max(map(len,xs)); out=torch.full((len(xs),n),value,dtype=torch.long); mask=torch.zeros_like(out)
    for i,x in enumerate(xs): out[i,-len(x):]=torch.tensor(x); mask[i,-len(x):]=1
    return out,mask

def _pad_right(xs,value):
    n=max(map(len,xs)); out=torch.full((len(xs),n),value,dtype=torch.long); mask=torch.zeros_like(out)
    for i,x in enumerate(xs): out[i,:len(x)]=torch.tensor(x); mask[i,:len(x)]=1
    return out,mask

class OfflineReplayTrainer(OneStepOffRayTrainer):
    def _init_async_rollout_manager(self):
        self.async_rollout_mode=False; self.async_rollout_manager=None; self.llm_server_manager=None
    def init_workers(self):
        self._init_resource_pools(); self._create_worker_classes(); self._init_worker_groups(); self._init_models()
        self.reward_loop_manager=None
    def _fit_update_weights(self): return None
    async def _async_gen_next_batch(self, continuous_iterator):
        try: epoch, raw=next(continuous_iterator)
        except StopIteration: return None
        groups=raw['groups']; prompts=[]; responses=[]; rewards=[]; uids=[]; sources=[]; gts=[]
        for g in groups:
            for item in g['items']:
                messages=[{'role':'system','content':ORCHESTRA_XML_SYSTEM_PROMPT},{'role':'user','content':build_orchestra_user_prompt(item['question'],'hotpotqa')}]
                p=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True)
                if isinstance(p,dict): p=p['input_ids']
                if hasattr(p,'tolist'): p=p.tolist()
                if p and isinstance(p[0],list): p=p[0]
                r=self.tokenizer.encode(item['response'],add_special_tokens=False)
                prompts.append(p); responses.append(r); rewards.append(float(item['replay_reward'])); uids.append(g['group_id']); sources.append('mas_gate_hotpotqa_offline_additive'); gts.append(item['ground_truth'])
        pp,pm=_pad_left(prompts,self.tokenizer.pad_token_id or 0); rr,rm=_pad_right(responses,self.tokenizer.pad_token_id or 0)
        inp=torch.cat([pp,rr],dim=1); att=torch.cat([pm,rm],dim=1); pos=torch.arange(inp.shape[1]).unsqueeze(0).expand(inp.shape[0],-1).clone(); pos[att==0]=0
        td=TensorDict({'prompts':pp,'responses':rr,'input_ids':inp,'attention_mask':att,'position_ids':pos},batch_size=[len(prompts)])
        b=DataProto(batch=td,non_tensor_batch={'uid':np.array(uids,dtype=object),'data_source':np.array(sources,dtype=object),'reward_model':np.array([{'ground_truth':x} for x in gts],dtype=object),'raw_reward':np.array(rewards,dtype=np.float32),'prompt_len':pm.sum(1).numpy(),'response_len':rm.sum(1).numpy()})
        b.batch['response_mask']=rm; b.meta_info['global_steps']=self.global_steps
        return {'offline':True}, {}, epoch, b, None
    def _fit_compute_reward(self,batch):
        r=torch.tensor(batch.non_tensor_batch['raw_reward'],dtype=torch.float32,device=batch.batch['responses'].device)
        mask=batch.batch['response_mask']; scores=torch.zeros_like(mask,dtype=torch.float32); lens=mask.sum(1).long(); scores[torch.arange(len(r),device=scores.device),torch.clamp(lens-1,min=0)]=r
        self.reward_tensor=scores; self.reward_extra_infos_dict={'offline_additive_reward':r.detach().cpu().numpy()}; return batch
    def _fit_dump_data(self,batch): return None
    def _fit_validate(self): return None
    def _fit_save_checkpoint(self):
        # parent policy saves final/intermediate checkpoints; keep inherited behavior
        return super()._fit_save_checkpoint()

def main():
    import hydra, ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    @ray.remote(num_cpus=10)
    class Runner:
        def run(self,config):
            OmegaConf.resolve(config); role_worker_mapping, cls=create_role_worker_mapping(config); validate_config(config=config,use_reference_policy=need_reference_policy(config),use_critic=need_critic(config)); local=copy_to_local(config.actor_rollout_ref.model.path,use_shm=False); tok=hf_tokenizer(local,trust_remote_code=config.data.get('trust_remote_code',False)); rpm=create_resource_pool_manager(config,role_worker_mapping.keys()); ds=ReplayDataset(config.data.train_files[0]); val=ReplayDataset(config.data.val_files[0]); tr=OfflineReplayTrainer(config,tok,role_worker_mapping,rpm,cls,train_dataset=ds,val_dataset=val,collate_fn=collate_groups,train_sampler=None,device_name=config.trainer.device); tr.init_workers(); asyncio.run(tr.fit())
    run_ppo(config,task_runner_class=Runner)
if __name__=='__main__': main()
