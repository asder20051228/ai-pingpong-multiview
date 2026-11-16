# 4090 NVDEC修复与自动单/双卡执行报告

## 结论

本机NVDEC已通过进程级兼容处理恢复，15段视频强制硬件解码全部通过。保持冻结86命中算法，在官方入口全量重测。下面均为公开开发集的本机结果，不是比赛隐藏测试集或榜单成绩。

| 实验 | TP/FP/FN | 准确率分/75 | 官方FPS | 推理秒 | 准备秒 | 丢失原86命中 |
|---|---|---:|---:|---:|---:|---|
| branch_diagnostic | 86/1/6 | 71.3854 | 42.4 | 49.8 | 0.9 | [] |
| fallback_control | 86/1/6 | 71.3854 | 35.2 | 59.9 | 2.1 | [] |
| nvdec_frozen_1 | 86/1/6 | 71.3854 | 38.4 | 55.0 | 1.9 | [] |
| nvdec_frozen_2 | 86/1/6 | 71.3854 | 37.9 | 55.6 | 2.2 | [] |
| nvdec_frozen_3 | 86/1/6 | 71.3854 | 38.1 | 55.4 | 2.0 | [] |
| package_cold | 86/1/6 | 71.3854 | 43.5 | 48.5 | 289.0 | [] |
| package_warm | 86/1/6 | 71.3854 | 45.8 | 46.0 | 0.7 | [] |
| pipeline_fallback | 86/1/6 | 71.3854 | 44.7 | 47.1 | 0.8 | [] |
| pipeline_nvdec_1 | 86/1/6 | 71.3854 | 41.7 | 50.5 | 0.8 | [] |
| pipeline_nvdec_2 | 86/1/6 | 71.3854 | 44.2 | 47.7 | 0.8 | [] |
| pipeline_nvdec_3 | 86/1/6 | 71.3854 | 45.8 | 46.0 | 0.7 | [] |

### 重复测速

- nvdec_frozen_: [38.4, 37.9, 38.1] FPS，中位数 **38.1 FPS**；1种预测文件SHA256。
- pipeline_nvdec_: [41.7, 44.2, 45.8] FPS，中位数 **44.2 FPS**；1种预测文件SHA256。

按赛题文档fps=1得1分、fps=50得25分的线性公式，44.2 FPS对应速度分约22.1592/25，加上71.3854准确率分，本机单卡估算合计93.5446/100。这不是官方双卡隐藏测试成绩。

fallback_control：冻结算法＋FFmpeg静默CPU回退。pipeline_fallback：异步流水线＋同一CPU回退。nvdec_frozen：冻结算法＋真实NVDEC。pipeline_nvdec：异步流水线＋真实NVDEC。branch_diagnostic：一张真实GPU上的双线程分支诊断，不能当作双卡测速。package_warm/package_cold：整理后的participant.solution:Solution入口，分别使用已有缓存及完全从ONNX构建。

## NVDEC故障证据和修复

- 驱动580.105.08，CUDA上下文创建成功，原cuvidGetDecoderCaps返回100（CUDA_ERROR_NO_DEVICE）。
- 宿主机RM枚举8张GPU，容器实际只获分配/dev/nvidia0。兼容库日志记录8→1过滤后，同一查询返回0，supported=1，num_nvdecs=1，max_width=4096。
- forced_nvdec_all.json：全部15段用h264_cuvid强制解码返回0且stderr为空，排除软解码回退。
- official_decode_pixels.json：正式FFmpeg GPU命令无stderr，首段完整125帧；RGB字节与CPU经NV12转换的结果完全相同。
- gpu.csv记录真实解码器利用率；CPU回退对照的decoder利用率为0。
- 修复使用局部LD_PRELOAD，未更换宿主驱动或系统库、未升级基础组件、未重启服务器。直接裸运行且不启用兼容启动器时，该580驱动故障依然存在。

相关原始记录：[NVIDIA容器工具问题1249](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1249)、[问题1282](https://github.com/NVIDIA/k8s-device-plugin/issues/1282)、[兼容库源码](https://github.com/flexgrip/nvidia-gpu-enumeration)。兼容库按上游GPL-3.0提供源码和许可证，位于upstream_nvenc_fix.c及upstream_LICENSE。

本机可用启动方式：

```bash
cd /root/autodl-tmp/pingpang_4090_20261018/nvdec_parallel_20261018
bash with_nvdec.sh /root/autodl-tmp/pingpang_4090_20261018/env/bin/python /root/autodl-tmp/pingpang_4090_20261018/precision_rescue_20261018/nvdec_probe.py
bash run_package.sh my_unique_run warm
```

with_nvdec.sh先探测NVDEC；仅遇到错误100时为该命令启用兼容库，并复查是否成功。已有正确驱动的环境无需兼容库。

## 比赛规则核对

本地赛题详情.md第152行明确指定Intel CPU + 2×4090，方案最多使用两张4090。第154行禁止修改core/、run.py、validate.py；第155行禁止升级替换基础组件。文档没有禁止参赛者模块内部多线程或多GPU并行，因此本方案将并行限制在participant代码内。

规则没有明确授予驱动调用兼容层的提交许可；服务器兼容处理单独保存，**没有自动加入参赛代码包**。正式比赛环境若NVDEC正常，只需参赛代码。若评测机同样存在驱动故障，应由平台修复或明确允许此环境处理。

## 代码改造

- submission/pingpang/participant/solution.py为可移植正式入口；所有权重和依赖模块在participant内，移除了实验服务器绝对路径依赖。
- torch.cuda.device_count()识别CUDA可见GPU，遵守CUDA_VISIBLE_DEVICES。0卡报明确错误；1卡用GPU0；≥2卡最多用GPU0和GPU1。
- 主线程继续官方解码，单个有序工作线程处理帧；最多2帧排队，复制帧缓冲防止复用污染。finish_video等待全部工作结束，计时完整包含队列处理。
- 双卡时：GPU0负责球模型与边界FP32复核，GPU1负责球台模型。两分支并发完成后严格按原帧顺序运行冻结轨迹、门控、颜色与去重逻辑。
- 没有按视频ID、帧号或GT选择策略，没有再改阈值。
- execution_tests.json覆盖0/1/2/8卡选择、最多2卡、有序处理、缓冲所有权、队列上限、视频重置、异步异常传播。
- 当前只有1张实际4090。双分支并发已在同卡模式全量诊断，真实两卡的数值、显存与加速比尚未测试。

## 冻结与复现

原冻结清单复核997文件，变化列表：[]。每轮正式入口前后均执行官方完整性校验。所有结果包含15视频、2109帧，详细run_status、scores、prediction SHA256见results.json和各运行目录。

准确率采用规则规定的逐视角F1×25相加、忽略无效GT口径。速度取官方infer_fps，准备和引擎构建时间另列。这里只给真实单卡速度，不将其冒充指定双卡平台的最终性能分。
