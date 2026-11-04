import time,statistics,gc
from vllm_gr.entrypoints.beam_search_v1.input_processor import BeamSearchInputProcessor
from vllm_gr.entrypoints.beam_search_v1.output_processor import BeamSearchOutputProcessor,_prompt_boundaries
from vllm_gr.entrypoints.beam_search_v1.protocol import OfflineBeamSearchRequest
from vllm_gr.v1.beam.batch_contracts import (BeamBatchOutput,BeamBatchResult,BeamOutputSequence,BeamRequestOutput,validate_beam_batch_request)
from vllm.v1.request import Request

def run(llm,prompt,params):
 call=OfflineBeamSearchRequest([{'prompt_token_ids':prompt}],params)
 processor=BeamSearchInputProcessor(llm.llm_engine)
 accepted=processor.process_offline_inputs(llm,call)
 native=accepted.items[0].request;hasher=llm.llm_engine.engine_core.engine_core.request_block_hasher
 output=llm.beam_search_v1([{'prompt_token_ids':prompt}],params,concurrency_limit=1)[0]
 sequences=tuple(BeamOutputSequence(tuple(x.token_ids[1:-1]),float(x.cumulative_logprob),(),x.finish_reason,None) for x in output.sequences)
 result=BeamBatchResult(accepted.protocol_version,accepted.batch_id,BeamBatchOutput((BeamRequestOutput(0,native.request_id,sequences),)),None)
 converter=BeamSearchOutputProcessor(llm.get_tokenizer(),boundary_token_ids=processor.boundary_token_ids)
 jobs={'frontend_request':lambda:processor.process_offline_inputs(llm,call),'native_request_and_hash':lambda:Request.from_engine_core_request(native,hasher),'validate_contract_once':lambda:validate_beam_batch_request(accepted),'prompt_boundaries':lambda:_prompt_boundaries(llm.get_tokenizer(),accepted,params,processor.boundary_token_ids),'compact_output':lambda:converter.process_outputs(call,accepted,result)}
 report={}
 for name,fn in jobs.items():
  for _ in range(20):fn()
  samples=[]
  for _ in range(7):
   previous=gc.isenabled();gc.disable();start=time.perf_counter_ns()
   for _ in range(1000):fn()
   elapsed=time.perf_counter_ns()-start
   if previous:gc.enable()
   samples.append(elapsed/1000/1000)
  report[name]={'median_us':statistics.median(samples),'round_average_us':samples,'n_per_round':1000,'rounds':7,'scope':'CPU-only actual function, no timing wrappers/profiler; initialization and inputs outside timer; nested rows overlap'}
 return report
