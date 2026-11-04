# 实验记录

主报告：[daily 对比报告](../grpc_daily_comparison_20261104.md)。

- `recif/`：真实数据、原版 daily CLI 的四轮交错对比；含原始结果、实际命令、数据校验和启动器。
- `fixed-1k/`：固定真实请求的补充归因；含六轮结果、原始命令（B3 由 run_one.py 启动）、CPU/GPU 诊断脚本及与 daily 原版 SHA 一致的脚本快照。

启动器保留实际机器路径，属于溯源记录，移机需要调整路径和运行库。归档不包含模型、Parquet、完整日志、编译产物、源码快照和固定 prompt token 内容。固定 prompt 可用相同数据和 tokenizer 按 worker.py 中的原始逻辑重新构造；摘要记录在 result.json 中。无需下载真实数据即可查看所有性能结果，但执行实验需另行准备数据和模型。

固定请求实验的 worker.py 在补充运行前增加了 isolated CPU 和 ablation 阶段；各轮具体采集项以 result.json 字段为准。所有 canonical 数据均在诊断和消融之前采集。
