# AMDGPU Notebook 更新教程

当前 Notebook 的源文件是 `scripts/build_amdgpu_notebook.py`，生成结果是
`notebooks/amdgpu_qwen25_7b.ipynb`。长期维护时改前者，再重新生成后提交两者；直接改 `.ipynb` 只适合临时试验。

## 在本地仓库更新

在 PowerShell 中执行：

```powershell
cd E:\DCU\CS-Competition
git switch feat/t4-triton-kernels
git pull --ff-only origin feat/t4-triton-kernels

# 修改 scripts/build_amdgpu_notebook.py 后重新生成
.venv\Scripts\python scripts\build_amdgpu_notebook.py

# 检查 JSON 和 Python 语法，再提交
.venv\Scripts\python -c "import json; json.load(open('notebooks/amdgpu_qwen25_7b.ipynb', encoding='utf-8')); print('notebook JSON OK')"
git diff --check
git add scripts/build_amdgpu_notebook.py notebooks/amdgpu_qwen25_7b.ipynb
git commit -m "Update AMDGPU experiment notebook"
git push origin feat/t4-triton-kernels
```

生成脚本中每个 `markdown("""...""")` 或 `code("""...""")` 就是一格。要加一格，通常把它放在相关实验前后：

```python
markdown("""
## 5. LongBench 小样本
""")

code("""
from pathlib import Path
DATA = Path("/mnt/data/dataset_staging")
print(DATA)
""")
```

不要把完整数据集、模型权重或 `results/*.json` 写进 Notebook。Notebook 只保留可复现命令和少量展示结果。

## 在 DCU Notebook 中同步仓库

Notebook 的仓库单元第一次运行时会 clone；以后再次运行会执行快进更新：

```python
!git -C /mnt/data/CS-Competition status --short
!git -C /mnt/data/CS-Competition pull --ff-only origin feat/t4-triton-kernels
!git -C /mnt/data/CS-Competition log -1 --oneline
```

如果 `pull` 报本地修改冲突，先保存需要的修改，或在确认没有重要修改后执行：

```python
!git -C /mnt/data/CS-Competition stash push -m notebook-local
!git -C /mnt/data/CS-Competition pull --ff-only origin feat/t4-triton-kernels
```

更新代码后，重新运行导入、模型加载和基准测试单元。旧的变量和旧模型可能仍在内存中，比较新旧方案前建议选择 **Kernel -> Restart Kernel**，再从头运行。

## 只想临时改一格

在 JupyterLab 中打开 `notebooks/amdgpu_qwen25_7b.ipynb`，点击 **Insert** 添加 Markdown 或 Code 单元，执行并确认结果即可。临时修改不会自动回写生成脚本；要保留它，应把同样内容复制到 `scripts/build_amdgpu_notebook.py`，重新生成并提交。

## 提交前检查

- Notebook 能从第一格按顺序运行到目标实验。
- `MODEL_DIR`、数据路径和输出目录是当前 DCU 环境真实存在的路径。
- 结果文件包含 dtype、attention、batch、上下文长度和 Git commit。
- 没有提交模型权重、数据集压缩包、缓存或本地虚拟环境。
