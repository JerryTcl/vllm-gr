import os,sys,json,importlib.util,hashlib,traceback,cProfile,pstats
from pathlib import Path
from transformers import AutoConfig,AutoTokenizer
from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams
from probes import Probes
root=Path(__file__).resolve().parent;label=sys.argv[1];out=root/label;out.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('daily',root/'run_offline_benchmark.py');daily=importlib.util.module_from_spec(spec);spec.loader.exec_module(daily)
model=os.environ['DAILY_AB_MODEL'];tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True);mc=AutoConfig.from_pretrained(model,trust_remote_code=True)
if not (root/'prompt.json').exists():
 from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset
 ds=OneRecDataset(dataset_path='/workspace/ws/data/grpc-daily-recif-20261104/A1/data',task_types=['video'],tokenizer=tok,custom_input_len=1023,sample_size=100)
 request=ds.sample(num_requests=100,tokenizer=tok,no_oversample=True)[0]
 ids=tok.encode(request.prompt,add_special_tokens=False);assert len(ids)==1023
 (root/'prompt.json').write_text(json.dumps({'token_ids':ids,'length':len(ids),'source':'first daily selected RecIF request'},indent=2))
prompt=json.loads((root/'prompt.json').read_text())['token_ids'];assert len(prompt)==1023
config,resource=daily.build_v1_constraint_config(catalog=Path('tools/AB_Tests/constraints/video_constraint_triples.json'),tokenizer=tok,model_config=mc,artifact=out/'constraint-table.safetensors',beam_width=128)
kwargs=daily.build_engine_kwargs(model=model,beam_width=128,max_model_len=40960,beam_api='v1',catalog=Path('tools/AB_Tests/constraints/video_constraint_triples.json'),additional_config={},v1_config=config,batch_size=1)
(out/'engine_kwargs.json').write_text(json.dumps(kwargs,indent=2))
params=BeamSearchParams(beam_width=128,max_tokens=5,temperature=0.0);params.begin_token='<|sid_begin|>';params.end_token='<|sid_end|>';params.ignore_eos=True
result={'commit':sys.argv[2],'prompt_sha256':hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest(),'canonical':[],'gpu_diagnostic':[]}
def save():(out/'result.json').write_text(json.dumps(result,indent=2))
with GRLLM(**kwargs) as llm:
 for _ in range(10):
  assert llm.reset_prefix_cache();daily.measure_request_canonical(llm,[prompt],params,'v1');daily.measure_request_canonical(llm,[prompt],params,'v1')
 for _ in range(100):
  assert llm.reset_prefix_cache();result['canonical'].append(daily.measure_request_canonical(llm,[prompt],params,'v1'));daily.measure_request_canonical(llm,[prompt],params,'v1')
 result['canonical_distribution']=daily.distribution_ms([x['e2e_ms'] for x in result['canonical']]);save();print('CANONICAL',json.dumps(result['canonical_distribution']),flush=True)
 try:
  from isolate_cpu import run as isolate_cpu
  result['isolated_cpu']=isolate_cpu(llm,prompt,params)
 except Exception:result['isolated_cpu_error']=traceback.format_exc()
 save();print('ISOLATED_CPU_DONE',flush=True)
 probe=Probes()
 try:
  probe.install(llm)
  for _ in range(30):
   assert llm.reset_prefix_cache();daily.measure_request_canonical(llm,[prompt],params,'v1')
  result['cpu_diagnostic']=probe.summary(30)
 except Exception:result['cpu_diagnostic_error']=traceback.format_exc()
 finally:probe.uninstall()
 save();print('CPU_DIAGNOSTIC_DONE',flush=True)
 try:
  for _ in range(10):
   assert llm.reset_prefix_cache();result['gpu_diagnostic'].append(daily.measure_request_diagnostic(llm,[prompt],params,'v1'))
 except Exception:result['gpu_diagnostic_error']=traceback.format_exc()
 save();print('GPU_DIAGNOSTIC_DONE',flush=True)
 # cProfile is a separate diagnostic and does not supply canonical latency.
 profile=cProfile.Profile()
 try:
  for _ in range(10):
   assert llm.reset_prefix_cache();profile.enable();daily.measure_request_canonical(llm,[prompt],params,'v1');profile.disable()
  profile.dump_stats(str(out/'cpu.pstats'))
  rows=[{'file':k[0],'line':k[1],'function':k[2],'calls_per_request':v[1]/10,'self_us_per_request':v[2]*1e6/10,'inclusive_us_per_request':v[3]*1e6/10} for k,v in pstats.Stats(profile).stats.items()]
  (out/'profile.json').write_text(json.dumps(sorted(rows,key=lambda x:-x['inclusive_us_per_request']),indent=2))
 except Exception:profile.disable();result['profile_error']=traceback.format_exc()
 save()
 if label=='B3':
  try:
   from ablation import run as run_ablation
   result['ablation']=run_ablation(llm,prompt,params,daily)
  except Exception:result['ablation_error']=traceback.format_exc()
  save()
print('DONE',label,flush=True)
