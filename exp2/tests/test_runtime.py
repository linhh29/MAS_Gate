import json,math,unittest
from gate_main.editor import Action,ActionKind,GateState,Graph,Node,validate_action,legal_actions
from gate_main.runtime import execute_edit
from gate_main.prompts import initial_messages,next_gate_message
from gate_main.training import node_quantities,Counterfactual,paired_reward
from gate_baselines.protocol import Graph as BaselineGraph

class Worker:
 def __init__(self):self.calls=[]
 def run(self,node,task,upstream,guidance='',fixed_prompt=None):
  self.calls.append((node.id,upstream.copy(),guidance,fixed_prompt));return node.id+' result',fixed_prompt or 'new prompt'

class RuntimeTests(unittest.TestCase):
 def state(self):
  g=Graph({'a':Node('a','retrieve',task='find evidence'),'b':Node('b','verify',parents=['a']),'z':Node('z','answer',parents=['b'],is_final=True)})
  return GateState('question',g,'b',{'a':'raw a','b':'raw b'},{'a':'saved a','b':'saved b'})
 def test_retry_reuses_prompt(self):
  state=self.state();worker=Worker();out=execute_edit(state,Action(ActionKind.RETRY_CURRENT,'b'),worker)
  self.assertEqual(worker.calls[0][3],'saved b');self.assertEqual(out.extra_calls,1)
 def test_backtrack_budget_and_safety(self):
  state=self.state();state.extra_calls_left=1
  with self.assertRaises(ValueError):validate_action(state,Action(ActionKind.BACKTRACK_PARENT,'a'))
  state=self.state();state.graph.nodes['b'].replay_safe=False;self.assertNotIn(ActionKind.RETRY_CURRENT,legal_actions(state));self.assertNotIn(ActionKind.BACKTRACK_PARENT,legal_actions(state))
 def test_annotation_and_insertion_preserve_context(self):
  s=self.state();w=Worker();payload={'error_type':'INCOMPLETE','target_span':'raw b','diagnosis':'Missing relation','downstream_check':'Verify relation'}
  result=execute_edit(s,Action(ActionKind.ANNOTATE_CURRENT,'b',payload),w);self.assertTrue(result.outputs['b'].startswith('raw b'));self.assertIn('[EDITOR ANNOTATION]',w.calls[-1][1]['b']);self.assertEqual(s.outputs['b'],'raw b')
  result=execute_edit(s,Action(ActionKind.INSERT_SUBTASK,'z',{'mode':'VERIFY','task':''}),Worker());self.assertIn('ORIGINAL MILESTONE:\nraw b',result.outputs['runtime_insert_b_z'])
 def test_payload_rejects_replacement_and_unobserved_quote(self):
  with self.assertRaises(ValueError):Action.parse('{"action":"KEEP","reason":"extra field"}')
  with self.assertRaises(ValueError):validate_action(self.state(),Action(ActionKind.ANNOTATE_CURRENT,'b',{'error_type':'EVIDENCE','target_span':'unseen','diagnosis':'check','downstream_check':'verify'}))
 def test_observation_and_turns(self):
  s=self.state();messages=initial_messages(s);obs=json.loads(messages[-1]['content'].split('OBSERVATION:\n')[-1]);self.assertEqual(set(obs['observed_outputs']),{'a','b'});self.assertEqual(obs['current_milestone_contract']['role'],'verify');self.assertNotIn('saved a',json.dumps(messages));self.assertNotIn('answer',obs)
  self.assertIn('current_output',next_gate_message(s)['content'])
 def test_additive_math_and_reward(self):
  q=node_quantities({'a':.5,'b':-.5,'z':.5},{'a':['b'],'b':['z'],'z':[]},{('a','b'):.8,('b','z'):.6});self.assertAlmostEqual(q['a']['propagation'],1.2);self.assertEqual(q['z']['propagation'],0);self.assertAlmostEqual(q['z']['impact'],math.tanh(.5))
  base=Counterfactual(Action(ActionKind.KEEP),0,.1);edited=Counterfactual(Action(ActionKind.RETRY_CURRENT,'a'),1,.3);self.assertAlmostEqual(paired_reward(base,edited),.94);self.assertEqual(paired_reward(base,base),0)
if __name__=='__main__':unittest.main()
