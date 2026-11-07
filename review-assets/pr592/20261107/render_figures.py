"""Regenerate figures and HTML from trace_summary.json (requires matplotlib)."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ROOT = OUT = Path(__file__).resolve().parent
summary = json.loads((ROOT / 'trace_summary.json').read_text())
selected = {variant: next(r for r in summary['rows'] if all(r[k] == v for k,v in ref.items())) for variant,ref in summary['representatives'].items()}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10, 'axes.spines.top':False,'axes.spines.right':False,
    'axes.spines.left':False,'axes.edgecolor':'#CBD5E1','axes.labelcolor':'#334155','xtick.color':'#475569','ytick.color':'#334155',
    'svg.fonttype':'none'})
COLORS = {'request':'#cbd5e1','add_requests':'#64748b','apply_staged_writes':'#94a3b8','all_token_ids':'#94a3b8',
    'uva':'#ea580c','prepare_inputs':'#0891b2','prepare_attn':'#0891b2','prefill_launch':'#4f46e5',
    'gpu_prefill':'#2563eb','gpu_decode':'#7c3aed'}

def timeline(zoom=False):
    fig, axes = plt.subplots(2,1,figsize=(13.0,7.0), dpi=150)
    fig.subplots_adjust(left=.205,right=.975,top=.93,bottom=.09,hspace=.5)
    keys = (['add_requests','apply_staged_writes','all_token_ids','uva','prepare_inputs','prepare_attn','prefill_launch','gpu_prefill'] if zoom
            else ['request','add_requests','uva','prepare_inputs','prepare_attn','prefill_launch','gpu_prefill','gpu_decode'])
    labels={'request':'CPU: offline request','add_requests':'CPU: add_requests','apply_staged_writes':'CPU: apply_staged_writes',
        'all_token_ids':'CPU: all_token_ids write','uva':'CPU: UVA token conversion','prepare_inputs':'CPU: prepare_inputs',
        'prepare_attn':'CPU: prepare_attn','prefill_launch':'CPU: Prefill graph launch','gpu_prefill':'GPU: Prefill graph span','gpu_decode':'GPU: Decode graph spans'}
    xmax = 36 if zoom else 240
    for ax,variant in zip(axes, ('parent','fixed')):
        row=selected[variant]; spans=row['spans']; origin=spans['add_requests' if zoom else 'request']['ts']
        uva=spans['uva']; ux=(uva['ts']-origin)/1000; ud=uva['dur']/1000
        ax.axvspan(ux,ux+ud,color='#fff7ed',zorder=-1)
        for y,key in enumerate(keys):
            entries=spans[key] if isinstance(spans[key],list) else [spans[key]]
            for e in entries:
                x=(e['ts']-origin)/1000; duration=e['dur']/1000
                ax.broken_barh([(x,duration)], (y-.29,.58), facecolors=COLORS[key], zorder=2)
                if key=='uva':
                    ax.annotate(f"{duration:.3f} ms  |  array() x {row['array_calls']:,}",xy=(x+duration,y),
                        xytext=(7,0),textcoords='offset points',ha='left',va='center',fontsize=10,fontweight='bold',color='#9a3412')
                elif zoom and key=='gpu_prefill':
                    ax.text(x+0.4,y,'continues >',va='center',ha='left',fontsize=9,color='white',clip_on=True)
        name='Before / parent-equivalent V2' if variant=='parent' else 'After / PR #592 V2'
        ax.set_title(f"{name}  -  block {row['block']}, request {row['request']}",loc='left',fontsize=12,fontweight='bold',pad=13)
        ax.set_yticks(range(len(keys)),[labels[k] for k in keys]);ax.set_ylim(len(keys)-.4,-.6);ax.set_xlim(0,xmax)
        ax.tick_params(axis='y',length=0,pad=10); ax.xaxis.set_major_locator(MultipleLocator(5 if zoom else 40))
        ax.grid(axis='x',color='#e2e8f0',linewidth=.7);ax.set_axisbelow(True)
        ax.set_xlabel('Time from add_requests entry (ms)' if zoom else 'Time from profiled request entry (ms)')
    file=OUT/('staging_zoom.svg' if zoom else 'request_timeline.svg')
    fig.savefig(file,facecolor='white');plt.close(fig)

timeline(); timeline(True)

table=''
for key,label in [('add_requests','Prefill add_requests'),('apply_staged_writes','RequestState.apply_staged_writes'),('all_token_ids','StagedWriteTensor.all_token_ids'),('uva','UVA token conversion')]:
    a=summary['statistics']['parent'][key]; b=summary['statistics']['fixed'][key]
    table+=f"<tr><td>{label}</td><td>{a['median_ms']:.3f} ms <small>[{a['min_ms']:.3f}, {a['max_ms']:.3f}]</small></td><td>{b['median_ms']:.3f} ms <small>[{b['min_ms']:.3f}, {b['max_ms']:.3f}]</small></td></tr>"

style='''body{margin:0;padding:36px 46px;background:#f1f5f9;font:16px/1.5 Arial,sans-serif;color:#172554}*{box-sizing:border-box}.page{width:1408px;margin:auto;background:white;padding:30px 34px;border:1px solid #dbe3ef;border-radius:12px}.tag{font-size:13px;letter-spacing:1.4px;color:#64748b;font-weight:700}h1{font-size:29px;line-height:1.25;margin:8px 0 8px}p{margin:9px 0;color:#475569}.meta{font-size:14px;color:#64748b}.plot{display:block;width:100%;height:auto}.note{background:#f8fafc;border-left:4px solid #3b82f6;padding:12px 16px;font-size:14px}.metrics{display:flex;gap:16px;margin:18px 0}.metric{flex:1;background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;padding:12px 18px}.metric b{display:block;font-size:25px;line-height:1.4;color:#172554}.metric span{font-size:13px;color:#64748b}table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}td,th{text-align:left;padding:9px 12px;border-bottom:1px solid #e2e8f0}th{background:#f8fafc;color:#334155}small{color:#64748b;font-size:12px}.foot{margin-top:14px;font-size:12px;color:#64748b}'''
ua=summary['statistics']['parent']['uva']['median_ms'];ub=summary['statistics']['fixed']['uva']['median_ms']
header='<div class="tag">PR #592 / TRACE EVIDENCE / 2026-11-07</div>'
meta='<p class="meta">OneRec-1.7B BF16 | L20 | B=4 | beam=128 | input=1024 | FULL Prefill + Decode backbones<br>Fix: c281cadc8576 | parent: 447fb736091d | vLLM 0.30.0 | NumPy 2.2.6</p>'
footer='<p class="foot">Paired collection: before / after / after / before; 10 warmups + 3 traced requests per block. Six samples per variant.<br>Representative request: nearest to each variant\'s median UVA slice duration. All outputs exact; FULL graph checks passed.</p>'
overview=header+'<h1>The fix removes a CPU delay before Prefill graph replay</h1>'+meta
overview+='<p>Both panels use the same 0-240 ms scale. Bars are timestamped events from the recorded traces.</p><img class="plot" src="request_timeline.svg">'
overview+='<div class="note"><b>Read the orange slice:</b> repeated NumPy concatenation delays request registration and subsequent graph submission. Blue/purple GPU ranges show device execution attributed to each graph launch.</div>'
overview+='<p class="foot">Profiled timelines are diagnostic samples, not benchmark means. The separate unprofiled B=4 benchmark was 210.783 → 186.912 ms (-11.33%, 200 samples/variant).</p>'+footer
detail=header+'<h1>UVA token conversion: 4,097 materializations become 1</h1>'+meta
detail+=f'<div class="metrics"><div class="metric"><span>BEFORE: median traced UVA slice</span><b>{ua:.3f} ms</b></div><div class="metric"><span>AFTER: median traced UVA slice</span><b>{ub:.3f} ms</b></div><div class="metric"><span>array() calls per B=4 request batch</span><b>4,097 → 1</b></div></div>'
detail+='<img class="plot" src="staging_zoom.svg"><table><tr><th>Nested CPU span</th><th>Before: median [min, max]</th><th>After: median [min, max]</th></tr>'+table+'</table>'
detail+='<div class="note">The 0-36 ms time axis is identical in both panels. The after slice is &lt;0.1 ms and therefore very narrow.<br>Nested durations overlap: do not sum these rows. Trace slices include profiler and instrumentation overhead.</div>'+footer
for name,content in [('overview',overview),('detail',detail)]:
    (OUT/f'{name}.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>PR #592 trace comparison</title><style>{style}</style><body><main class="page">{content}</main></body></html>')

summary['trace_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.pt.trace.json.gz')}
(OUT/'trace_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'statistics':summary['statistics'],'representatives':summary['representatives']},indent=2))
