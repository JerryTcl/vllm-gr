import functools,time,sys,importlib,collections
class Probes:
 def __init__(self):self.restore=[];self.events=collections.defaultdict(list)
 def wrap(self,original,name):
  @functools.wraps(original)
  def timed(*a,**k):
   wall=time.perf_counter_ns();cpu=time.thread_time_ns()
   try:return original(*a,**k)
   finally:self.events[name].append((time.perf_counter_ns()-wall,time.thread_time_ns()-cpu))
  return timed
 def attr(self,obj,name,tag):
  if not hasattr(obj,name):return
  original=getattr(obj,name);self.restore.append((obj,name,original));setattr(obj,name,self.wrap(original,tag))
 def function(self,module,name,tag):
  if not hasattr(module,name):return
  original=getattr(module,name);wrapped=self.wrap(original,tag)
  for key,mod in list(sys.modules.items()):
   if key.startswith('vllm_gr.') and mod is not None:
    for attr,value in list(vars(mod).items()):
     if value is original:self.restore.append((mod,attr,value));setattr(mod,attr,wrapped)
 def install(self,llm):
  from vllm_gr.entrypoints.beam_search_v1.client import BeamSearchClient
  for method,tag in [('prepare_request','frontend.prepare'),('submit_once','core.submit'),('wait_final','engine.wait'),('build_output','frontend.output'),('close','session.close')]:self.attr(BeamSearchClient,method,tag)
  from vllm_gr.entrypoints.beam_search_v1 import input_processor as ip,output_processor as op
  self.attr(ip.BeamSearchInputProcessor,'process_offline_inputs','input.process_offline')
  self.function(ip,'_create_batch_request','input.build_batch')
  self.function(op,'_prompt_boundaries','output.prompt_boundaries')
  from vllm_gr.v1.beam import batch_contracts
  self.function(batch_contracts,'validate_beam_batch_request','validation.batch_contract')
  from vllm_gr.v1.engine import beam_search_v1,core,gr_async_scheduler
  self.function(beam_search_v1,'validate_gr_request','validation.gr_request')
  self.function(core,'_admit_gr_session','core.admit_session')
  self.function(gr_async_scheduler,'_prepare_gr_session','scheduler.admit')
  from vllm_gr.v1.worker import gr_execution,beam_final_output
  self.function(gr_execution,'worker_input','worker.detach_inputs')
  self.attr(beam_final_output.GPUBeamFinalOutput,'consume','worker.final_output_consume')
  from vllm.v1.request import Request
  self.attr(Request,'__init__','request.native_init')
  self.attr(Request,'update_block_hashes','request.update_hashes')
  core_obj=llm.llm_engine.engine_core.engine_core
  self.attr(core_obj,'preprocess_add_request','core.preprocess_add')
  self.attr(core_obj,'request_block_hasher','hash.request_blocks')
  if hasattr(core_obj.scheduler,'schedule'):self.attr(core_obj.scheduler,'schedule','scheduler.schedule')
  try:
   from vllm_gr.v1.beam import token_buffer
   self.function(token_buffer,'initialize_numpy_request','request.numpy_adapt')
  except ImportError:pass
  from vllm.v1.engine.input_processor import InputProcessor
  self.attr(InputProcessor,'process_inputs','input.native_processor')
  from vllm_gr.entrypoints.beam_search_v1.output_processor import BeamSearchOutputProcessor
  self.attr(BeamSearchOutputProcessor,'process_outputs','output.process')
 def uninstall(self):
  for obj,name,old in reversed(self.restore):setattr(obj,name,old)
 def summary(self,n):
  return {name:{'calls_per_request':len(values)/n,'wall_us_per_request':sum(x[0] for x in values)/n/1000,'cpu_us_per_request':sum(x[1] for x in values)/n/1000,'inclusive':True} for name,values in self.events.items()}
