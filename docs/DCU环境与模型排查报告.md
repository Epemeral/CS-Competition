# DCU 环境与模型排查报告

## 一、排查结论

截至当前排查，已经获得超算互联网华中一区 A 区的 Slurm 账号，并确认登录节点、共享文件系统和 DCU 作业示例可访问，但**尚未发现或确认赛题指定的 `Qwen2.5-7B-Instruct` 权重目录**。

当前也无法列出“有哪些可用模型”，因为已有终端记录只有文件系统顶层目录，没有任何模型目录、`config.json` 或权重文件的有效搜索结果。此前对 `/public` 的全局搜索因范围过大被手动中断，不能据此判断模型不存在。

因此，目前没有证据表明环境里存在符合赛题要求的模型，也没有证据表明模型一定未部署。需要组委会或指导老师提供模型及比赛环境的准确入口。

## 二、已确认的环境信息

| 项目 | 当前结果 | 说明 |
|---|---|---|
| 登录节点 | `zz-login02` | 公告明确要求仅用于编辑和编译，不应直接运行推理 |
| 用户目录 | `/public/home/xdzs2026_b0028` | 位于共享 NFS |
| 主要共享存储 | `/public`，约 9.2 PB | 根目录含数万个条目，不适合无边界递归搜索 |
| 其他共享存储 | `/slurm_share`、`/work5` | `/work5` 约 1.6 PB，同样需要准确路径 |
| 本地数据盘 | `/data`，约 3.5 TB | 当前主要是系统日志，不像比赛模型目录 |
| 示例分区 | `huge` | 来自通用 `dcutest.slurm`，尚未确认是否为本次比赛分区 |
| 示例资源 | 4 节点、每节点 4 块 DCU | 不适合直接照搬到单卡 7B 推理任务 |
| 示例软件 | `compiler/rocm/2.9` | 很旧的通用 HPL 示例，不能代表比赛提供的 PyTorch/DTK 环境 |

## 三、已发现的可疑公共目录

以下目录名称与 AI、软件或 DCU 有关，但现有记录未显示其中包含哪个模型：

- `/public/ai_data`
- `/public/DL_DATA`
- `/public/dtk`
- `/public/SothisAI`
- `/public/appmarket`
- `/public/software`
- `/work5`

`/public/share` 下约有 5.5 万个目录，可能包含其他用户数据，不应进行深层全局扫描。

## 四、符合赛题模型的判定标准

候选目录必须对应 `Qwen2.5-7B-Instruct`，而不是旧版 Qwen、其他参数规模或基础模型。至少检查：

1. 存在 `config.json`、tokenizer 配置、generation 配置和完整权重分片；
2. `config.json` 中 `model_type` 通常为 `qwen2`；
3. `architectures` 通常包含 `Qwen2ForCausalLM`；
4. 模型规模与 7B 配置相符，权重总体积通常约 15 GB（FP16/BF16，具体分片数可不同）；
5. 模型名称或模型卡明确为 `Qwen2.5-7B-Instruct`；
6. 能在组委会指定的统一容器/环境中离线加载。

仅看到 `Qwen`、`Qwen-7B`、`Qwen2-7B`、`Qwen2.5-7B` 或某个 `config.json`，都不足以确认符合要求。

## 五、下一步排查方法

仓库提供了只读脚本 `scripts/inventory_models.sh`。它只搜索几个最可能的公共目录，并限制深度和时间，不会扫描整个 `/public`：

```bash
cd ~/CS-Competition
bash scripts/inventory_models.sh model_inventory.txt
sed -n '1,260p' model_inventory.txt
```

若代码尚未上传到集群，也可以先由老师或助教直接提供绝对路径，避免低效搜索。

## 六、需要指导老师或组委会确认的问题

1. 赛题预置的 `Qwen2.5-7B-Instruct` 权重绝对路径是什么？
2. 比赛是否提供统一容器镜像？镜像路径和启动命令是什么？
3. 指定的 DTK、PyTorch、Transformers 和 vLLM 版本分别是什么？
4. 应使用哪个 Slurm partition、account/QOS，以及单卡 DCU 的申请参数？
5. 官方 `baseline.py`、`evaluate.py` 和三套评测数据集从哪里获取？
6. 登录节点、计算节点是否允许访问 Hugging Face/ModelScope？若模型未预置，应下载到哪个团队目录？
7. 团队是否有独立的共享项目目录和容量配额？

## 七、可直接发送给指导老师的说明

老师您好，我们已经登录超算互联网华中一区 A 区账号，确认用户目录为 `/public/home/xdzs2026_b0028`，并查看了平台提供的 DCU Slurm 示例。目前示例使用 `huge` 分区和 `compiler/rocm/2.9`，看起来是通用 HPL 示例，尚不能确认是本次赛题环境。

我们检查了 `/public`、`/work5` 等共享挂载信息，但 `/public` 下目录数量很大，全局搜索会长时间占用登录节点。当前尚未找到或确认 `Qwen2.5-7B-Instruct` 的预置权重、比赛容器、官方 baseline/evaluate 脚本和评测集。

烦请协助确认以下信息：模型权重的绝对路径；比赛容器及启动方式；指定的 DTK/PyTorch/vLLM 版本；应使用的 Slurm 分区、QOS 和单卡申请参数；官方代码及评测集的获取位置。如果模型尚未部署，也请告知推荐的下载位置和团队存储目录。谢谢！
