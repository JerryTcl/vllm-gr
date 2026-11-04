import os,json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
env=dict(os.environ,**json.loads(Path('/workspace/ws/data/grpc-daily-recif-20261104/env.json').read_text()))
env['DAILY_AB_MODEL']='/workspace/ws/data/vllm-gr-models/hf-cache/hub/models--OpenOneRec--OneRec-1.7B/snapshots/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b';env['PYTHONPATH']=str(r/'source-B')
out=r/'B3';out.mkdir(exist_ok=True)
print('START B3 isolated CPU and controlled ablation',flush=True)
with (out/'run.log').open('w') as f:
 p=subprocess.run(['/workspace/ws/data/beam-framework-bench/je-env/bin/python',str(r/'worker.py'),'B3','ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60'],cwd=r/'source-B',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
print('DONE B3',p.returncode,flush=True)
raise SystemExit(p.returncode)
