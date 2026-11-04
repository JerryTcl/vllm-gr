import os,json,subprocess,tarfile,io,hashlib,time,random
from pathlib import Path
root=Path(__file__).resolve().parent;repo=Path('/workspace/ws/data/vllm-gr-grpc-binary');dashboard=Path('/workspace/ws/data/vllm-gr-performance-dashboard');data=Path('/workspace/ws/data/video_test.parquet')
commits={'A':'33a1fdb2e39dea9e30f648405bd263eb62b15323','B':'ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60'}
for side,commit in commits.items():
 src=root/('source-'+side);src.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit],cwd=repo))) as tar:tar.extractall(src)
script=dashboard/'tools/daily_benchmark/run_offline_benchmark.py'
provenance={'commits':commits,'dashboard_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=dashboard,text=True).strip(),'script':str(script),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'model_revision':'cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b','hardware':'H20','warmup_requests':10,'measured_pairs':100,'dataset':str(data),'scope':'Canonical miss only; unmodified daily CLI, posthoc diagnostic disabled; hit retained by daily paired protocol.'}
(root/'provenance.json').write_text(json.dumps(provenance,indent=2))
print('Prepared snapshots; waiting for complete Parquet footer',flush=True)
while True:
 try:
  with data.open('rb') as f:f.seek(-4,2);ready=f.read(4)==b'PAR1'
  if ready:break
 except (OSError,ValueError):pass
 time.sleep(5)
import pandas as pd
h=hashlib.sha256()
with data.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
sha=h.hexdigest();frame=pd.read_parquet(data,columns=['metadata'])
indices=[str(i) for i in frame.sample(n=100,random_state=42).index];random.Random(0).shuffle(indices)
ids=['video_'+i for i in indices];ids_hash=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
expected_sha='66079915e3c3daab7df3487f9babb658df99ec1672c3ef42d7893d87cef2e6d8';expected_ids='48ed79ffcb501b982ec4343816e45477264c362781eb0881a066cf45be0f1afa'
metadata={'file_sha256':sha,'expected_file_sha256':expected_sha,'file_matches_published_daily':sha==expected_sha,'sample_ids_sha256':ids_hash,'sample_ids_match_published_daily':ids_hash==expected_ids,'rows':len(frame),'bytes':data.stat().st_size,'sample_ids':ids}
(root/'dataset-verification.json').write_text(json.dumps(metadata,indent=2));print('DATASET',json.dumps({k:v for k,v in metadata.items() if k!='sample_ids'}),flush=True)
if sha!=expected_sha or ids_hash!=expected_ids:raise SystemExit('Dataset verification mismatch: inspect dataset-verification.json before benchmarking')
env=dict(os.environ,**json.loads(Path('/workspace/ws/data/grpc-daily-aligned-20261104/env.json').read_text()))
model=env.pop('DAILY_AB_MODEL')
(root/'env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('VLLM','OMP','HF','CUDA','LD_LIBRARY','LIBRARY_PATH'))},indent=2))
for label in ['A1','B1','B2','A2']:
 src=root/('source-'+label[0]);out=root/label;out.mkdir(exist_ok=True)
 data_dir=out/'data';(data_dir/'video').mkdir(exist_ok=True,parents=True)
 target=data_dir/'video/video_test.parquet'
 if not target.exists():target.symlink_to(data)
 env['PYTHONPATH']=str(src)
 cmd=['/workspace/ws/data/beam-framework-bench/je-env/bin/python',str(script),'--output',str(out/'raw-result.json'),'--data-dir',str(data_dir),'--catalog',str(src/'tools/AB_Tests/constraints/video_constraint_triples.json'),'--model',model,'--num-prompts','100','--warmup-requests','10','--diagnostic-prompts','0','--beam-api','v1','--beam-width','128','--input-length','1024','--batch-size','1']
 (out/'command.json').write_text(json.dumps(cmd,indent=2));print('START',label,flush=True)
 with (out/'run.log').open('w') as f:p=subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 print('DONE',label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
