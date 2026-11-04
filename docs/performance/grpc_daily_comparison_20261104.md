# gRPC / NumPy 合入前后：daily 配方离线性能对比（2026-11-04）

使用与 daily 相同的真实 RecIF Parquet 和100条源数据抽样，对比 gRPC NumPy 合入前后的 B1、内部输入1024、beam128、生成3步的 miss API 耗时。只统计 miss；hit 调用保留原脚本协议，不作性能分析。

本地 H20 上，真实数据两轮交错测试观察到 miss 均值 **+0.184 ms（+0.48%）**。固定单请求补充对比隔离 #542 后为 **+0.116 ms**；CPU 定位显示额外成本主要在输入转换、重复校验和输出转换。两组实验分别统计，不混合均值。

这里测试的是合入 gRPC / NumPy 代码后的 **offline Python API**，不包含 gRPC 服务、网络或序列化往返；两版 API 输入均为 Python list。

## 数据和环境对齐

| 项目 | 对齐情况 |
| --- | --- |
| 原始 Parquet | SHA256与 daily 相同：`66079915e3c3daab7df3487f9babb658df99ec1672c3ef42d7893d87cef2e6d8` |
| 数据规模 | 38781行，405017893 bytes |
| 100个抽样ID及顺序 | 摘要与 daily 相同：`48ed79ffcb501b982ec4343816e45477264c362781eb0881a066cf45be0f1afa` |
| 输入长度 | API均为1023 tokens，追加 SID begin 后内部1024 |
| 方法和参数 | 直接运行 daily 原版 Python CLI，10组预热、100对 miss/hit，B1、beam128、max_tokens5、实际生成3步 |
| 硬件 | 本地H20，历史daily为L20，不完全相同 |
| 模型 | 本地OneRec 1.7B固定revision `cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b`；历史daily未记录模型revision |
| tokenizer指纹 | 本地 `sha256:00119100fcd12439532a0530a884baa2b5641c362a02be3bff7aeba38a716023`，历史为 `sha256:fc1a1519ffec0c8606438ff011ef76ad3be32e0e0cf17c4f2079d0e4125dbe78`，不相同 |

原始数据和源行抽样已完全对齐；分词后的token序列、模型版本和运行环境不能声称与历史daily完全一致。本次本地前后两版使用相同模型、tokenizer、约束表和环境，比较的是同一机器上的commit差值。

## Commit 和测量方法

| 版本 | Commit |
| --- | --- |
| base | `33a1fdb2e39dea9e30f648405bd263eb62b15323` |
| gRPC NumPy 合入后 | `ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60` |

使用 git archive 的干净源码，不含本地前端 prompt list 复用补丁。此commit对同时跨过 #552 和 #542，不能单独把全部差值归因于NumPy。

原版脚本：[tools/daily_benchmark/run_offline_benchmark.py](https://github.com/DINGEde/vllm-gr-performance-dashboard/blob/1c322fd629f4269a050fccc884214944181286d2/tools/daily_benchmark/run_offline_benchmark.py)。看板commit `1c322fd629f4269a050fccc884214944181286d2`；脚本SHA256 `f607d9c0feec9490ff704288f5f53be7973b5fbe2f56c637733f1576e41acc01`，运行结束已重新校验脚本未变。

- 原版配置：V1 ModelRunner、进程内Core、max_num_seqs=1、max_logprobs=128、max_model_len及max_num_batched_tokens均40960、FCFS、async scheduling、prefix cache开启、chunked prefill关闭、GPU memory utilization=0.5。
- 原版完整video constraint triples、CUDA constraint table、beam graph、worker_decision、short-suffix attention开启，prefill graph KV bound=4096。
- 预热10组，采用已发布daily记录的值；工具包当前默认值为4。随后每组在计时外 reset KV prefix cache，计时miss，再立即重复输入执行hit。
- canonical计时从直接调用beam_search_v1之前到torch.cuda.synchronize之后；默认compact输出，输出消费在计时外，没有内部profiler或计时monkeypatch；posthoc diagnostic设0。
- A1、B1、B2、A2交错运行，每版两次独立engine、共200个miss样本，全部请求成功。每轮单独的数据目录只链接原始Parquet，避免继承其它运行的sample缓存。
- local OMP_NUM_THREADS=8、OMP_WAIT_POLICY=passive；两版相同，不声称与历史L20容器的未记录CPU环境相同。

## Miss 性能

| 版本 | 第一轮均值 ms | 第二轮均值 ms | 合并均值 ms | P50 ms | P90 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 38.373 | 38.264 | 38.319 | 38.276 | 38.568 | 38.669 |
| gRPC NumPy 合入后 | 38.529 | 38.477 | 38.503 | 38.452 | 38.738 | 38.868 |

合并均值变化 **+0.184 ms，+0.48%**；P90变化 +0.170 ms。具体轮次和全部分布保留在原始JSON中。

历史L20真实数据的miss均值为51.936到54.087 ms，增加2.151 ms；本地H20结果与其不是同一次完全相同环境的复现，尤其不能忽略tokenizer指纹不同。

## 证据

- [汇总和逐轮统计](./grpc-daily-20261104/recif/comparison.json)
- [数据文件与抽样校验](./grpc-daily-20261104/recif/dataset-verification.json)
- [运行入口](./grpc-daily-20261104/recif/run.py)
- [commit和原版脚本溯源](./grpc-daily-20261104/recif/provenance.json)
- [运行环境](./grpc-daily-20261104/recif/env.json)

各轮原版命令及 raw-result.json 已随报告保存在 `grpc-daily-20261104/recif/{A1,B1,B2,A2}/`。完整日志、权重、原始 Parquet 和编译约束资源仅保留在原测试机器，未提交。


## 输入数据的具体处理

本次使用用户提供的 `/workspace/ws/data/video_test.parquet`，不是随机 token 或 mock 输入。原始 Parquet 和抽样源行 ID/顺序均通过与已发布 daily 的摘要比对。

由 vllm-gr 的 `benchmarks/open_one_rec/open_one_rec_dataset.py::OneRecDataset` 处理 video 数据；100 行按 pandas `sample(n=100, random_state=42)` 抽样，再按种子 0 打乱。daily CLI 使用 `custom_input_len=1023` 构造 prompt，在计时前执行 `tokenizer.encode(..., add_special_tokens=False)`。因此本次是源自真实数据、按 daily 方法规整到 1K 的输入，并非自然变长输入测试。

每轮使用独立 data 目录，并将原始文件链接为 `data/video/video_test.parquet`，避免已有 sample 缓存干扰。原始数据的 ID 摘要相同不等于不同 tokenizer 编码后的 token 摘要相同；本地 A/B 的 tokenizer 一致，和历史 daily 的差异已在上表披露。

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

## 固定 1K 请求的补充归因

为定位小幅差值，固定上述真实数据抽样中的第一条 prompt，继续使用 daily 配置和 `measure_request_canonical`。每进程预热 10 组、测量 100 组，仍然每组 reset→miss→hit。使用干净源码，无工作区 list 复用补丁。

| 版本 | commit | 第一次均值 ms | 第二次均值 ms | 合并均值 ms |
|---|---|---:|---:|---:|
| #552 only | `d2b91452fb8144f3e3cf4cbf9127df1b9ecda132` | 38.1509 | 38.3275 | 38.2392 |
| #552 + #542 | `ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60` | 38.3463 | 38.3650 | 38.3556 |

这组隔离 #542 的观测差值为 **+0.1164 ms（+0.30%）**。#552 组两轮均值自身相差 0.1765 ms，因此不把 API 差值当作精确的因果分解。

prefix cache 开启，但每次 reset 清 KV cache，仅统计 miss。固定输入反复执行后，CPU hash memo 是热的；清 KV cache 不会清除此 memo。这与此前冷 hash CPU 微测的 +0.12 ms 不是同一种场景，不能叠加。

在 canonical 之后，另测真实 CPU 函数，7 轮×1000 次，取轮均值中位数；计时段没有 GPU 或 profiler：

| 场景 | 每请求频次 | #552 only µs | +#542 µs | 增量 |
|---|---:|---:|---:|---:|
| 前端构造请求，含 list→NumPy | 1 | 100.09 | 134.74 | +34.66 µs |
| 数组契约检查 | 5 | 1.81/次 | 3.87/次 | +2.06 µs/次 |
| 输出 `_prompt_boundaries`，含 NumPy→list | 1 | 2.18 | 13.51 | +11.33 µs |
| 整个 compact 输出转换 | 1 | 97.22 | 110.14 | +12.93 µs |
| native Request 构造及热 hash | 1 | 15.43 | 11.88 | −3.55 µs |

这些行存在包含关系：前端构造包含一次契约检查，compact 输出包含 `_prompt_boundaries`，不能全部相加。独立 CPU 测量能解释约 50–60 µs 的新增工作。canonical 之后的全链路分段探针观察到约 90 µs 增量，但带探针观测开销，仅用于定位。

入口新增 `np.asarray(..., dtype=np.int64)`；契约检查每请求运行 5 次，包含 dtype/维度/连续性/非空/`tokens.min()` 检查；默认 compact 输出也调用 `_prompt_boundaries` 将 prompt 数组转回 list。固定请求下，native Request 与热 hash 未见退化；GPU event 诊断没有足够证据认定 kernel 本身退化。

同进程实验性复用输出边界后约减少 11 µs，与独立 CPU 测量一致；临时跳过数组 guard 的消融存在约 0.095 ms 首尾漂移，不用它宣称 guard 的精确 API 成本。没有把删除校验当作正式修复。

daily 的进程内 Core 配方仍承担 list API 的转换成本，不能获得跨进程/网络的 NumPy 序列化收益。剩余差值尚未完全闭合；本报告不把整个 0.1–0.3 ms 区间全部归因到单一函数。

固定请求的 [汇总 JSON](./grpc-daily-20261104/fixed-1k/comparison.json)、各轮 result.json 及诊断脚本随报告归档。该部分启动器调用的是 daily 的配置构造及 canonical 函数，并在其后另采 CPU/GPU 诊断；与主报告直接运行原版 CLI 的方式有所区别。
