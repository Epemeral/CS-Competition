# CS-Competition

先导杯（HNU 校内赛）赛题一：Qwen2.5-7B-Instruct 在海光 DCU 上的推理系统优化。

当前仓库处于 DCU 账号发放前的准备阶段。`baseline.py --tiny` 只用于本地验证工程链路，不能代表官方成绩。

## 快速开始

```powershell
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements-local.txt
.venv/Scripts/python baseline.py --tiny --output results/tiny-run.json
.venv/Scripts/python baseline.py --tiny --batch-size 4 --length-bucket --output results/t1-length.json
.venv/Scripts/python -m unittest discover -s tests -v
```

## 目录

- `baseline.py`：准备版 PyTorch/Transformers 推理与测量入口。
- `evaluate.py`：token 级正确性比较。
- `t1/core.py`：与模型无关的校验逻辑。
- `kernels/`：T4 阶段的 Triton 融合算子工作台（参考实现 → kernel → 正确性验证 → 性能基准）。
  独立于 T1，可在 Colab / DCU 集群上单独运行，详见 `kernels/README.md`。
- `scripts/qwen25_amdgpu_benchmark.py`：方式三 AMDGPU/ROCm 上的 Qwen2.5-7B Transformers 基线与吞吐基准。
- `scripts/inspect_datasets.py`：盘点外部 JSON/JSONL/Parquet 数据集的字段和样本规模。
- `scripts/convert_datasets.py`：将 LongBench、HelloBench、AX、MMLU-Pro、tau2 转成统一 JSONL 合约。
- `notebooks/amdgpu_qwen25_7b.ipynb`：ModelScope + AMDGPU Notebook 入口。
- `docs/T1学习指南.md`：概念和本地练习。
- `docs/T1代码解构.md`：执行链路和代码说明。
- `docs/T1优化分析.md`：T1 可验证优化、实验矩阵和 T1/T2/T3/T4 边界。
- `docs/T4实施计划.md`：T4 算子、融合、vLLM 接入和 DCU 验收路线。
- `docs/后续步骤.md`：DCU 账号到位后的路线。
- `docs/方式三AMDGPU Notebook指南.md`：方式三环境的安装、基线和优化顺序。
- `docs/真实DCU快速部署.md`：64 GiB 单卡真实环境的快速部署、批量扫描和结果判读。
- `docs/数据集接入计划.md`：五个外部数据集的用途、优先级和 Notebook 接入顺序。
- `docs/Notebook文件更新教程.md`：生成、同步、运行和提交 Notebook 的操作步骤。

拿到组委会镜像后，以官方 `baseline.py`、`evaluate.py`、模型路径和评测集为准，并把环境信息写入实验记录。

真实 DCU 节点可直接运行 `scripts/dcu_quickstart.sh`；脚本复用组委会提供的 PyTorch/DTK，不会覆盖底层运行时。
