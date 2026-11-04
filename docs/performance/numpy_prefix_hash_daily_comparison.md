# Daily 配方离线性能对比：list、NumPy 与 NumPy hash 优化

本文整理 2026-11-04 已完成的两组测试，复用已有结果，不重新运行测试。第一组为 gRPC/NumPy 合入前后的对比；第二组包含 NumPy hash 优化后的对比。两组结果分别统计。

本分支基于 main `55b9f42`，带入 NumPy hash 优化及对应测试。下文性能数据来自表中列出的测试 commit 与补丁，不是基于该 main 版本重新测量的结果。

## 共同测试方法

### 脚本

实际测量直接调用 performance-dashboard 的原版脚本：

- 仓库：[DINGEde/vllm-gr-performance-dashboard](https://github.com/DINGEde/vllm-gr-performance-dashboard)。
- commit：`1c322fd629f4269a050fccc884214944181286d2`。
- 脚本：[tools/daily_benchmark/run_offline_benchmark.py](https://github.com/DINGEde/vllm-gr-performance-dashboard/blob/1c322fd629f4269a050fccc884214944181286d2/tools/daily_benchmark/run_offline_benchmark.py)。
- SHA256：`f607d9c0feec9490ff704288f5f53be7973b5fbe2f56c637733f1576e41acc01`。
- [原版脚本快照](./numpy-prefix-hash-daily-20261104/run_offline_benchmark.py)。

各组 `run.py` 负责导出源码、准备数据目录、设置环境和按顺序启动 CLI；API 计时由原版 daily 脚本的 `measure_request_canonical()` 完成。

### 输入数据

| 项目 | 本次测试使用值 |
|---|---|
| 数据 | 用户提供的真实 RecIF video 数据 `/workspace/ws/data/video_test.parquet` |
| 规模 | 38781 行，405017893 bytes |
| Parquet SHA256 | `66079915e3c3daab7df3487f9babb658df99ec1672c3ef42d7893d87cef2e6d8`，与 daily 数据摘要相同 |
| 抽样 | pandas `sample(n=100, random_state=42)`，再以 seed 0 打乱 |
| 100 条源行 ID 及顺序摘要 | `48ed79ffcb501b982ec4343816e45477264c362781eb0881a066cf45be0f1afa`，与 daily 摘要相同 |
| 输入长度 | API 输入 1023 tokens，追加 SID begin 后内部 1024 tokens |
| API 输入类型 | 所有版本均传 Python list；NumPy 版本在 GR 请求边界转为 ndarray |

使用 vllm-gr 的 `benchmarks/open_one_rec/open_one_rec_dataset.py::OneRecDataset` 处理 video 数据，`custom_input_len=1023`；daily CLI 在计时前执行 `tokenizer.encode(..., add_special_tokens=False)`。输入来自真实数据，并按 daily 方法规整到 1K。每轮使用独立 data 目录，将原始文件链接为 `data/video/video_test.parquet`。

### 模型、环境与配置

| 项目 | 设置 |
|---|---|
| GPU / CPU | H20 / Xeon Platinum 8457C |
| 模型 | OneRec 1.7B，BF16，使用真实权重 |
| 模型 revision | `cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b` |
| tokenizer 摘要 | `sha256:00119100fcd12439532a0530a884baa2b5641c362a02be3bff7aeba38a716023` |
| 运行库 | 同一 venv，vLLM 0.22.1，PyTorch 2.11 + CUDA 13.0 |
| API | `GRLLM.beam_search_v1`，默认 compact 输出 |
| batch / beam | 1 / 128 |
| 生成长度 | `max_tokens=5`，包含 SID 边界，实际生成 3 步 |
| Runner / Core | V1 ModelRunner；进程内 EngineCore，无前端到 Core 的 IPC |
| prefix cache | **开启** |
| 调度 | FCFS，async scheduling 开，chunked prefill 关 |
| 容量 | `max_num_seqs=1`，`max_logprobs=128`，`max_model_len=40960`，`max_num_batched_tokens=40960` |
| GPU memory utilization | 0.5 |
| 约束 | 完整 `tools/AB_Tests/constraints/video_constraint_triples.json`，CUDA constraint table |
| 执行选项 | beam graph、worker_decision、short-suffix attention 开启；prefill graph KV bound 4096 |
| CPU 环境 | `OMP_NUM_THREADS=8`，`OMP_WAIT_POLICY=passive` |

### 计时与统计口径

每轮独立启动 engine，先 10 组 warm-up，再测量 100 组请求。每组按以下顺序执行：

```text
reset KV prefix cache（计时外）
    → 调用 API 测 miss
    → 立即重复同一输入执行 hit
```

本文只统计 miss。hit 保留 daily 的执行顺序，不计入本文的性能均值。

计时从直接调用 `beam_search_v1` 之前开始，到 API 返回并执行 `torch.cuda.synchronize()` 之后结束。包含输入处理、调度、模型执行及 compact 输出构造；不包含 engine 初始化、数据读取、tokenizer 编码、reset 和返回后的输出消费。关闭内部探针与 profiler，`--diagnostic-prompts 0`。

KV cache reset 不清除 CPU hash memo；按上述预热和 miss/hit 顺序执行，不额外清理 hash memo。每版本两轮、合计 200 个 miss 样本；合并分位数根据两轮原始样本计算。

### 单轮 CLI

各轮完整实际命令均归档为 `command.json`。模板如下，四个路径变量分别指 Python、dashboard、对应源码和本地模型；每轮使用新的 DATA_DIR 与 RESULT：

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

CUDA/linker 等本机环境见两组各自的 `env.json`。归档启动器保留实际测试机器路径。

## 1. Daily 配方：list vs NumPy

直接运行上述原版 daily CLI。两版均使用 `git archive` 导出的干净源码，不含本地 prompt list 复用或 hash 优化补丁。

| 标记 | 版本 | 测试 commit |
|---|---|---|
| A | list / base | `33a1fdb2e39dea9e30f648405bd263eb62b15323` |
| B | gRPC/NumPy 合入后 | `ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60` |

版本范围包含 #552 和 #542。执行顺序 **A1 → B1 → B2 → A2**，共 400 个 miss 样本，全部成功。

| 版本 | 第一轮均值 ms | 第二轮均值 ms | 合并均值 ms | 合并 P50 ms | 合并 P90 ms |
|---|---:|---:|---:|---:|---:|
| list / base | 38.373272 | 38.263856 | **38.318564** | 38.275693 | 38.567880 |
| NumPy | 38.528624 | 38.476963 | **38.502794** | 38.451582 | 38.738080 |

NumPy 相对 list：均值 **+0.184230 ms（+0.48%）**，P90 **+0.170200 ms**。

归档：

- [启动器](./numpy-prefix-hash-daily-20261104/list-vs-numpy/run.py)
- [汇总统计](./numpy-prefix-hash-daily-20261104/list-vs-numpy/comparison.json)
- [数据校验](./numpy-prefix-hash-daily-20261104/list-vs-numpy/dataset-verification.json)
- [版本与脚本溯源](./numpy-prefix-hash-daily-20261104/list-vs-numpy/provenance.json)
- [运行环境](./numpy-prefix-hash-daily-20261104/list-vs-numpy/env.json)
- `list-vs-numpy/{A1,B1,B2,A2}/`：各轮 `command.json` 与 `raw-result.json`。

## 2. Daily 配方：list vs NumPy opt

仍直接运行同一原版 daily CLI，输入文件、100 条抽样、模型、tokenizer、约束表、环境、参数与计时方法同上。

| 标记 | 版本 | 测试 commit / 补丁 |
|---|---|---|
| A | list / #552 only | `d2b91452fb8144f3e3cf4cbf9127df1b9ecda132` |
| B | NumPy 未优化 | `ecccac6dab44dbbf38397b1c757aa9b7b6c7ed60` |
| N | NumPy opt | `ecccac6d` + 下方归档的 hash 补丁 |

N 与 B 仅 `vllm_gr/v1/engine/engine_core_patch.py` 不同，未带入 prompt list 复用改动。执行顺序 **A1 → B1 → N1 → N2 → B2 → A2**，共 600 个 miss 样本，全部成功。

| 版本 | 第一轮均值 ms | 第二轮均值 ms | 合并均值 ms | 合并 P50 ms | 合并 P90 ms |
|---|---:|---:|---:|---:|---:|
| list / #552 only | 38.344665 | 38.546824 | **38.445744** | 38.283423 | 38.951362 |
| NumPy 未优化 | 38.731840 | 38.715555 | **38.723698** | 38.565624 | 39.205201 |
| NumPy opt | 38.544250 | 38.571914 | **38.558082** | 38.469376 | 38.958944 |

| 对比 | 均值变化 ms | P90 变化 ms |
|---|---:|---:|
| NumPy 未优化 − list | +0.277954 | +0.253839 |
| **NumPy opt − list** | **+0.112338** | **+0.007582** |
| NumPy opt − NumPy 未优化 | −0.165616 | −0.246257 |

归档：

- [启动器](./numpy-prefix-hash-daily-20261104/list-vs-numpy-opt/run.py)
- [汇总统计](./numpy-prefix-hash-daily-20261104/list-vs-numpy-opt/comparison.json)
- [数据校验](./numpy-prefix-hash-daily-20261104/list-vs-numpy-opt/dataset-verification.json)
- [版本与脚本溯源](./numpy-prefix-hash-daily-20261104/list-vs-numpy-opt/provenance.json)
- [运行环境](./numpy-prefix-hash-daily-20261104/list-vs-numpy-opt/env.json)
- [实际测量的 hash 补丁](./numpy-prefix-hash-daily-20261104/list-vs-numpy-opt/hash-only.patch)，SHA256：`bacee4a4ed86d0f63cc7224d4f801a496ec908d049c234da5500e03d5c819be8`。
- `list-vs-numpy-opt/{A1,B1,N1,N2,B2,A2}/`：各轮 `command.json` 与 `raw-result.json`。

## 本分支的代码修改

NumPy block hash 从 `tuple(map(int, block))` 改为直接使用 int64 ndarray 的 `.tobytes()`；首次整链计算与后续增量计算采用同一格式。hash 接口按 GR 的 ndarray 输入处理，不保留 list/tuple 兼容分支。NumPy 的 block hash 值随之改变。

代码位于 [engine_core_patch.py](../../vllm_gr/v1/engine/engine_core_patch.py)，对应测试位于 [test_numpy_tokens.py](../../tests/test_numpy_tokens.py)。未带入其它本地优化。归档的性能数据及补丁保持原样；本分支随后简化了 hash 代码，删除类型判断、list/tuple 分支、额外格式标记及字节序转换，未重新测量性能。

本次只整理数据并将修改移到最新 main，没有重跑测试。此前旧测试环境中的相关 CPU 回归为 69 项通过；本分支基于 main 的测试尚未执行。归档不包含模型、原始 Parquet、完整日志和编译产物。
