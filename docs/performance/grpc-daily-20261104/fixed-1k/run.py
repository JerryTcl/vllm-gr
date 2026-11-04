import sys,runpy
from pathlib import Path
if '--extra-candidate' in sys.argv:
 runpy.run_path(str(Path(__file__).with_name('run_one.py')),run_name='__main__');raise SystemExit()
import os,json,subprocess,tarfile,io,shutil,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent;repo=Path('/workspace/ws/data/vllm-gr-grpc-binary');dash=Path('/workspace/ws/data/vllm-gr-performance-dashboard')
commits={'A':'33a1fdb2e39dea9e30f648405bd263eb62b15323','C':'d2b91452fb8144f3e3cf4cbf9127df1b9ecda132','B':'ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60'}
for side,commit in commits.items():
 src=root/('source-'+side);src.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit],cwd=repo))) as f:f.extractall(src)
script=dash/'tools/daily_benchmark/run_offline_benchmark.py';shutil.copy(script,root/script.name)
(root/'provenance.json').write_text(json.dumps({'commits':commits,'daily_script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'mode':'fixed first real RecIF prompt; daily kwargs and canonical timer; probes collected only after canonical samples'},indent=2))
env=dict(os.environ,**json.loads(Path('/workspace/ws/data/grpc-daily-recif-20261104/env.json').read_text()))
env['DAILY_AB_MODEL']='/workspace/ws/data/vllm-gr-models/hf-cache/hub/models--OpenOneRec--OneRec-1.7B/snapshots/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b'
for label in ['A1','C1','B1','B2','C2']:
 src=root/('source-'+label[0]);env['PYTHONPATH']=str(src);out=root/label;out.mkdir(exist_ok=True);print('START',label,flush=True)
 cmd=['/workspace/ws/data/beam-framework-bench/je-env/bin/python',str(root/'worker.py'),label,commits[label[0]]];(out/'command.json').write_text(json.dumps(cmd,indent=2))
 with (out/'run.log').open('w') as f:p=subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 print('DONE',label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
