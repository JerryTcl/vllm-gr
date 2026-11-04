import os,json,subprocess,tarfile,io,hashlib,shutil
from pathlib import Path
root=Path(__file__).parent;repo=Path('/workspace/ws/data/vllm-gr-grpc-binary');dashboard=Path('/workspace/ws/data/vllm-gr-performance-dashboard');data=Path('/workspace/ws/data/video_test.parquet')
commits={'A':'d2b91452fb8144f3e3cf4cbf9127df1b9ecda132','B':'ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60','N':'ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60'}
patch=subprocess.check_output(['git','diff','--','vllm_gr/v1/engine/engine_core_patch.py'],cwd=repo);(root/'hash-only.patch').write_bytes(patch)
for side,commit in commits.items():
 src=root/('source-'+side);src.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit],cwd=repo))) as tar:tar.extractall(src)
 if side=='N':subprocess.run(['git','apply',str(root/'hash-only.patch')],cwd=src,check=True)
script=dashboard/'tools/daily_benchmark/run_offline_benchmark.py'
shutil.copy(script,root/'daily-script-snapshot.py')
verification=json.loads(Path('/workspace/ws/data/grpc-daily-recif-20261104/dataset-verification.json').read_text())
assert hashlib.sha256(data.read_bytes()).hexdigest()==verification['file_sha256']
(root/'dataset-verification.json').write_text(json.dumps(verification,indent=2))
(root/'provenance.json').write_text(json.dumps({'commits':commits,'N_patch':'hash-only.patch','patch_sha256':hashlib.sha256(patch).hexdigest(),'dashboard_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=dashboard,text=True).strip(),'daily_script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'order':['A1','B1','N1','N2','B2','A2'],'scope':'Unmodified daily CLI; real RecIF data; prefix cache enabled; canonical miss only. No prompt list reuse patch; N differs from B only by hash-only.patch.'},indent=2))
env=dict(os.environ,**json.loads(Path('/workspace/ws/data/grpc-daily-recif-20261104/env.json').read_text()))
model='/workspace/ws/data/vllm-gr-models/hf-cache/hub/models--OpenOneRec--OneRec-1.7B/snapshots/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b'
(root/'env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('VLLM','OMP','HF','CUDA','LD_LIBRARY','LIBRARY_PATH'))},indent=2))
for label in ['A1','B1','N1','N2','B2','A2']:
 src=root/('source-'+label[0]);out=root/label;out.mkdir(exist_ok=True);data_dir=out/'data';(data_dir/'video').mkdir(exist_ok=True,parents=True)
 (data_dir/'video/video_test.parquet').symlink_to(data)
 env['PYTHONPATH']=str(src)
 cmd=['/workspace/ws/data/beam-framework-bench/je-env/bin/python',str(script),'--output',str(out/'raw-result.json'),'--data-dir',str(data_dir),'--catalog',str(src/'tools/AB_Tests/constraints/video_constraint_triples.json'),'--model',model,'--num-prompts','100','--warmup-requests','10','--diagnostic-prompts','0','--beam-api','v1','--beam-width','128','--input-length','1024','--batch-size','1']
 (out/'command.json').write_text(json.dumps(cmd,indent=2));print('START',label,flush=True)
 with (out/'run.log').open('w') as f:p=subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 print('DONE',label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
assert hashlib.sha256(script.read_bytes()).hexdigest()==json.loads((root/'provenance.json').read_text())['daily_script_sha256']
