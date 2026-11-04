# gRPC / NumPy 合入前后：daily 配方离线性能对比（2026-11-04）

使用与 daily 相同的真实 RecIF Parquet 和100条源数据抽样，对比 gRPC NumPy 合入前后的 B1、内部输入1024、beam128、生成3步的 miss API 耗时。只统计 miss；hit 调用保留原脚本协议，不作性能分析。

真实数据测试的 miss 合并均值为 38.319 → 38.503 ms，变化 **+0.184 ms（+0.48%）**。

这里测试的是合入 gRPC / NumPy 代码后的 **offline Python API**，不包含 gRPC 服务、网络或序列化往返；两版 API 输入均为 Python list。

## 输入数据和运行环境

| 项目 | 本次测试 |
| --- | --- |
| 原始 Parquet | SHA256与 daily 相同：`66079915e3c3daab7df3487f9babb658df99ec1672c3ef42d7893d87cef2e6d8` |
| 数据规模 | 38781行，405017893 bytes |
| 100个抽样ID及顺序 | 摘要与 daily 相同：`48ed79ffcb501b982ec4343816e45477264c362781eb0881a066cf45be0f1afa` |
| 输入长度 | API均为1023 tokens，追加 SID begin 后内部1024 |
| 方法和参数 | 直接运行 daily 原版 Python CLI，10组预热、100对 miss/hit，B1、beam128、max_tokens5、实际生成3步 |
| 硬件 | H20 |
| 模型 | OneRec 1.7B，固定 revision `cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b` |
| tokenizer指纹 | `sha256:00119100fcd12439532a0530a884baa2b5641c362a02be3bff7aeba38a716023` |

前后两版使用同一台机器、同一 venv、相同模型、tokenizer、约束表和配置。

## Commit 和测量方法

| 版本 | Commit |
| --- | --- |
| base | `33a1fdb2e39dea9e30f648405bd263eb62b15323` |
| gRPC NumPy 合入后 | `ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60` |

使用 git archive 的干净源码，不含本地前端 prompt list 复用补丁。版本范围包含 #552 和 #542。

原版脚本：[tools/daily_benchmark/run_offline_benchmark.py](https://github.com/DINGEde/vllm-gr-performance-dashboard/blob/1c322fd629f4269a050fccc884214944181286d2/tools/daily_benchmark/run_offline_benchmark.py)。看板commit `1c322fd629f4269a050fccc884214944181286d2`；脚本SHA256 `f607d9c0feec9490ff704288f5f53be7973b5fbe2f56c637733f1576e41acc01`，运行结束已重新校验脚本未变。

- 原版配置：V1 ModelRunner、进程内Core、max_num_seqs=1、max_logprobs=128、max_model_len及max_num_batched_tokens均40960、FCFS、async scheduling、prefix cache开启、chunked prefill关闭、GPU memory utilization=0.5。
- 原版完整video constraint triples、CUDA constraint table、beam graph、worker_decision、short-suffix attention开启，prefill graph KV bound=4096。
- 预热10组，采用已发布daily记录的值；工具包当前默认值为4。随后每组在计时外 reset KV prefix cache，计时miss，再立即重复输入执行hit。
- canonical计时从直接调用beam_search_v1之前到torch.cuda.synchronize之后；默认compact输出，输出消费在计时外，没有内部profiler或计时monkeypatch；posthoc diagnostic设0。
- A1、B1、B2、A2交错运行，每版两次独立engine、共200个miss样本，全部请求成功。每轮单独的数据目录只链接原始Parquet，避免继承其它运行的sample缓存。
- OMP_NUM_THREADS=8、OMP_WAIT_POLICY=passive，两版相同。

## Miss 性能

| 版本 | 第一轮均值 ms | 第二轮均值 ms | 合并均值 ms | P50 ms | P90 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 38.373 | 38.264 | 38.319 | 38.276 | 38.568 | 38.669 |
| gRPC NumPy 合入后 | 38.529 | 38.477 | 38.503 | 38.452 | 38.738 | 38.868 |

合并均值变化 **+0.184 ms，+0.48%**；P90变化 +0.170 ms。具体轮次和全部分布保留在原始JSON中。

## 证据

- [汇总和逐轮统计](./grpc-daily-20261104/recif/comparison.json)
- [数据文件与抽样校验](./grpc-daily-20261104/recif/dataset-verification.json)
- [运行入口](./grpc-daily-20261104/recif/run.py)
- [commit和原版脚本溯源](./grpc-daily-20261104/recif/provenance.json)
- [运行环境](./grpc-daily-20261104/recif/env.json)

各轮原版命令及 raw-result.json 已随报告保存在 `grpc-daily-20261104/recif/{A1,B1,B2,A2}/`。完整日志、权重、原始 Parquet 和编译约束资源仅保留在原测试机器，未提交。


## 输入数据的具体处理

本次使用用户提供的真实数据文件 `/workspace/ws/data/video_test.parquet`。原始 Parquet 和抽样源行 ID/顺序均通过与已发布 daily 的摘要比对。

由 vllm-gr 的 `benchmarks/open_one_rec/open_one_rec_dataset.py::OneRecDataset` 处理 video 数据；100 行按 pandas `sample(n=100, random_state=42)` 抽样，再按种子 0 打乱。daily CLI 使用 `custom_input_len=1023` 构造 prompt，在计时前执行 `tokenizer.encode(..., add_special_tokens=False)`。输入按 daily 方法规整到 1K。

每轮使用独立 data 目录，并将原始文件链接为 `data/video/video_test.parquet`，避免已有 sample 缓存干扰。

## 启动入口与复跑方式

实际 A/B 启动器为本报告附带的 [recif/run.py](./grpc-daily-20261104/recif/run.py)。它仅负责导出干净 commit、验证数据、设置环境并依次启动 A1→B1→B2→A2；实际测量直接调用 dashboard 的原版 CLI，没有另写 canonical 计时逻辑。

下面是单轮命令模板；四轮实际绝对路径命令见各目录 `command.json`。执行前需自行准备相同 Python 环境、模型、原始数据和该版本 dashboard。`DATA_DIR/video/video_test.parquet` 必须指向原始文件；每轮换新的 DATA_DIR 和 RESULT 路径。

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_GR_LIGHTWEIGHT_TIMING=0
export VLLM_GR_PREFILL_GRAPH_KV_BOUND=4096
export OMP_NUM_THREADS=8
export OMP_WAIT_POLICY=passive
export HF_HUB_OFFLINE=1
export PYTHONPATH="$SOURCE"
cd "$SOURCE"
"$PYTHON" "$DASHBOARD/tools/daily_benchmark/run_offline_benchmark.py" \
  --output "$RESULT" --data-dir "$DATA_DIR" \
  --catalog "$SOURCE/tools/AB_Tests/constraints/video_constraint_triples.json" \
  --model "$MODEL" --num-prompts 100 --warmup-requests 10 \
  --diagnostic-prompts 0 --beam-api v1 --beam-width 128 \
  --input-length 1024 --batch-size 1
```

`SOURCE` 为对应 commit 的干净源码目录；`MODEL` 为上述固定 revision 的本地模型目录。其余 CUDA/linker 路径是本机环境，实际记录见 [env.json](./grpc-daily-20261104/recif/env.json)，迁移机器时需使用本机匹配的运行库。归档的 `run.py` 保留原机器路径，是原始实验记录，未包装成通用安装工具。

## 固定 1K 请求的补充测试

固定上述真实数据抽样中的第一条 prompt，使用 daily 配置和 `measure_request_canonical`。每进程预热 10 组、测量 100 组，仍然每组 reset→miss→hit。使用干净源码，无工作区 list 复用补丁。

| 版本 | commit | 第一次均值 ms | 第二次均值 ms | 合并均值 ms |
|---|---|---:|---:|---:|
| #552 only | `d2b91452fb8144f3e3cf4cbf9127df1b9ecda132` | 38.1509 | 38.3275 | 38.2392 |
| #552 + #542 | `ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60` | 38.3463 | 38.3650 | 38.3556 |

合并均值变化 **+0.1164 ms（+0.30%）**。执行顺序为 base A1 → #552 C1 → #542 B1 → #542 B2 → #552 C2；表中统计 C1/C2 与 B1/B2。

prefix cache 开启，每次 reset 清 KV cache，仅统计 miss。固定输入重复执行，CPU hash memo 已热；reset 不清除此 memo。

该组启动器通过 worker.py 调用 daily 的配置构造函数和 canonical 计时函数。API 结果在所有 CPU/GPU 诊断及消融之前采集；补充 B3 不纳入上表。

固定请求的 [汇总 JSON](./grpc-daily-20261104/fixed-1k/comparison.json)、各轮 result.json 和实验脚本随报告归档。
