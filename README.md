# CS-Competition

先导杯（HNU 校内赛）赛题一：Qwen2.5-7B-Instruct 在海光 DCU 上的推理系统优化。

当前仓库处于 DCU 账号发放前的准备阶段。`baseline.py --tiny` 只用于本地验证工程链路，不能代表官方成绩。

## 快速开始

```powershell
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements-local.txt
.venv/Scripts/python baseline.py --tiny --output results/tiny-run.json
.venv/Scripts/python -m unittest discover -s tests -v
```

## 目录

- `baseline.py`：准备版 PyTorch/Transformers 推理与测量入口。
- `evaluate.py`：token 级正确性比较。
- `t1/core.py`：与模型无关的校验逻辑。
- `docs/T1学习指南.md`：概念和本地练习。
- `docs/T1代码解构.md`：执行链路和代码说明。
- `docs/后续步骤.md`：DCU 账号到位后的路线。

拿到组委会镜像后，以官方 `baseline.py`、`evaluate.py`、模型路径和评测集为准，并把环境信息写入实验记录。
