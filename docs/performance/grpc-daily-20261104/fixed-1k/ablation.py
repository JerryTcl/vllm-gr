import ast,inspect,sys,statistics
from vllm_gr.v1.beam import batch_contracts as bc
from vllm_gr.entrypoints.beam_search_v1 import output_processor as op

def replace_aliases(old,new):
 changed=[]
 for name,module in list(sys.modules.items()):
  if name.startswith('vllm_gr.') and module is not None:
   for key,value in list(vars(module).items()):
    if value is old:changed.append((module,key,value));setattr(module,key,new)
 return changed

def run(llm,prompt,params,daily):
 # This diagnostic is valid ONLY for this fixed, known-valid input. Production
 # source is not changed; the original function is restored before returning.
 assert len(prompt)==1023 and min(prompt)>=0
 source=inspect.getsource(bc.validate_beam_batch_request);tree=ast.parse(source);removed=[]
 class RemoveNumpyGuard(ast.NodeTransformer):
  def visit_If(self,node):
   if any(isinstance(x,ast.Constant) and x.value=='GR prompt_token_ids must be a nonempty contiguous nonnegative int64 array' for x in ast.walk(node)):
    removed.append(ast.unparse(node));return None
   return self.generic_visit(node)
 tree=RemoveNumpyGuard().visit(tree);ast.fix_missing_locations(tree);assert len(removed)==1
 namespace=dict(vars(bc));exec(compile(tree,'diagnostic_without_numpy_guard.py','exec'),namespace)
 fast=namespace['validate_beam_batch_request'];old=bc.validate_beam_batch_request
 report={'scope':'Fixed known-valid prompt only; experimental in-process guard removal, not a production implementation','removed_guard':removed[0],'samples':{}}
 def measure(name):
  values=[]
  for _ in range(3):
   assert llm.reset_prefix_cache();daily.measure_request_canonical(llm,[prompt],params,'v1');daily.measure_request_canonical(llm,[prompt],params,'v1')
  for _ in range(100):
   assert llm.reset_prefix_cache();values.append(daily.measure_request_canonical(llm,[prompt],params,'v1')['e2e_ms']);daily.measure_request_canonical(llm,[prompt],params,'v1')
  report['samples'][name]={'distribution':daily.distribution_ms(values),'values_ms':values}
 measure('normal_before')
 changed=replace_aliases(old,fast)
 original_boundaries=op._prompt_boundaries;cache=[]
 def cached_boundaries(*a,**k):
  if not cache:cache.append(original_boundaries(*a,**k))
  return cache[0]
 try:
  measure('without_new_numpy_guard')
  op._prompt_boundaries=cached_boundaries
  measure('without_guard_and_repeated_prompt_materialization')
 finally:
  op._prompt_boundaries=original_boundaries
  for obj,name,fn in reversed(changed):setattr(obj,name,fn)
 measure('normal_restored')
 return report
