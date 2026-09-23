# T2 Notebook 工作台

文件：`t2_optimization_workbench.ipynb`

生成或更新 Notebook：

```powershell
python notebooks/t2/build_t2_workbench_nb.py
```

在 Notebook 平台中上传 `notebooks/t2/t2_optimization_workbench.ipynb`，按单元格顺序运行。Notebook 会检查 Qwen2.5-7B 的 GQA 形状，比较 eager attention 与 PyTorch SDPA，测量两种 KV-cache 布局，并在 GPU 可用时记录中位数耗时。

结果默认保存到：

```text
results/t2/t2_results.json
```

`results/` 已被 Git 忽略，适合保存每次 DCU 实验的本地结果。提交报告时应同时记录镜像、PyTorch/ROCm、commit、warmup、重复次数和完整 JSON。

T2 的局部 attention 结果不能直接替代端到端比赛成绩。正式验收还要接回 T1 baseline，测 prefill/decode 的 token-level 一致性、TTFT、TPOT、吞吐、P50/P95 和峰值显存。
