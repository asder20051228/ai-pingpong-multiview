# 乒乓球多视角落点检测

2026 动感地带 AI＋高校创智计划 AI 技术赛道参赛工程。方案结合三帧球定位、球台区域预测、轨迹事件判定、颜色约束和 FP32 边界复核，输出触台事件的帧号与图像坐标。

采用已冻结的 `nvdec_parallel_20261018` 正式版本，保留完整官方框架和篮球参考实现。Docker 默认入口为 `/participant/run_all.sh`。

## 成果与报告

- 技术与作业过程报告（20 页）：[在线阅读](docs/technical-report.md) · [PDF](docs/technical-report.pdf) · [可编辑 Word](docs/technical-report.docx)
- [完整 Docker 镜像及公开验证视频](https://github.com/asder20051228/ai-pingpong-multiview/releases/tag/v2026.11.16)
- [本次容器验证记录](validation/container_validation_summary.json) · [历史实验依据](evidence/)

完整报告包含初始结果、模型训练与选择、逐轮精度改进和失败尝试，重点解释引擎缓存、解码回退排查、有序流水线及速度复测。新增 [过程证据索引](evidence/process_history/index.json) 保存 73 个历史文件的校验值；文中 23 项证据链接可直接追溯原始记录。

| 验证项目 | 结果 |
| --- | --- |
| 本机环境 | RTX 4060 Laptop GPU，Docker Desktop/WSL2，禁网容器 |
| 联合公开验证 | 乒乓球 15/15，篮球 2/2 视频成功 |
| 输出格式错误 / CPU 解码回退 | 0 / 0 |
| 乒乓球有效事件 | 86 TP，1 FP，6 FN（本地宽松口径） |
| 公开集精度分 | 宽松 71.3854/75；严格 70.3010/75 |
| 本机推理速度 | 39.1 FPS，仅作功能验证运行记录 |
| 历史单卡 RTX 4090 | 三次热启动中位数 45.9 FPS；宽松口径估算总分 94.38 |

公开验证集曾参与调优，以上不是官方隐藏测试成绩。指定 2×4090 环境的真实双卡性能仍需目标平台验证。完整实验设置与限制见技术报告。

## 获取完整 Docker 镜像

完整镜像归档约 **14.5 GB**，通过 GitHub Release 提供 14 个分卷。下载工具会并行下载、支持中断续传、校验每卷 SHA256、合并后校验完整镜像，成功后清理分卷缓存。首次下载合并建议预留至少 **30 GB** 磁盘空间，另需为 Docker 加载准备足够空间。

要求 Python 3.9 以上、curl 和 Docker。无需安装 Python 第三方依赖。

```bash
git clone https://github.com/asder20051228/ai-pingpong-multiview.git
cd ai-pingpong-multiview
python tools/download_image.py --load
```

若只下载镜像，去掉 `--load`。重复运行会复用已校验完成的分卷和归档。默认保存到 `downloads/`，也可用 `--output /path/to/storage` 指定目录。

手动合并时，按 `part01` 至 `part14` 的顺序二进制拼接；不要逐卷解压。完整归档 SHA256：

```text
f6fd04893d10c5a8394597ac3e2acf7de01d2738ea6909eb06b3102b801cf5aa
```

加载后镜像标签：`pingpang-submit:20261115`。镜像 ID：

```text
sha256:4c6aaebaa39d220407f84cdb5935f582edbccaebef8ec7d951094f5b4c0cd8c4
```

## 运行

需要支持 NVIDIA GPU 的 Linux Docker 或已配置 GPU 支持的 Docker Desktop。完整镜像架构为 `linux/amd64`，包含官方基础环境、两项任务、模型权重及公开验证视频。

```bash
mkdir -p /data/output
docker run --rm --gpus all --network none \
  -v /data/input:/participant/input:ro \
  -v /data/output:/participant/output \
  pingpang-submit:20261115
```

输入目录包含 `pingpang/`、`basketball/` 两个子目录，每个任务目录包含对应 manifest 与视频。以上为 Bash 命令；PowerShell 请写成一行或使用反引号续行。

使用镜像内置公开集执行环境、方案及联合自检：

```bash
mkdir -p "$PWD/test-output"
docker run --rm --gpus all --network none \
  --mount "type=bind,source=$PWD/test-output,target=/participant/output" \
  --entrypoint /selftest.sh pingpang-submit:20261115
```

首次运行会为当前 GPU 构建 TensorRT 引擎，准备阶段可能需要数分钟。镜像不包含特定 GPU 的 TensorRT 缓存。

## 从源码构建

代码仓库包含框架、参赛者代码和 ONNX 权重，验证视频由 Release 单独提供。首先下载视频，并加载组委会原始 `sport-base_v1.tar`：

```bash
python tools/download_data.py
docker load -i /path/to/sport-base_v1.tar
docker build --network none --pull=false -t pingpang-submit:20261115 .
```

随后使用上面的自检命令。官方框架代码保持原样；`.gitattributes` 保留 `participant/` 内文件的原始字节，避免 Windows 换行转换破坏完整性校验。

复核公开集预测：

```bash
python tools/score_public.py test-output/pingpang/predictions.jsonl --output local-scores.json
```

这个评分工具分别给出严格、宽松两种忽略事件处理口径，不替代官方隐藏评分器。

## 目录

```text
participant/    两项任务框架、参赛者代码、模型、manifest 与真值
tools/          镜像和验证视频下载、校验及本地评分工具
docs/           技术报告 PDF 与 Word
evidence/       历史实验、模型选择、原始预测和评分依据
validation/     当前容器测试与镜像完整性记录
release/        分卷下载清单、文件大小与 SHA256
Dockerfile      官方 sport-base:v1 上的构建定义
selftest.sh     环境、方案和联合入口自检
```

第三方资料来源见 [NOTICE.md](NOTICE.md)。
