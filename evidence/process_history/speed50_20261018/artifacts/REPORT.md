# 单卡4090的50 FPS优化实验

## 结果

本轮未达到全量50 FPS。保留原86命中冻结版本，所有新方案只在独立目录实验；没有将候选替换原正式版本。

| 方案 | 三次正式入口全量FPS | 中位数FPS | 范围 | TP/FP/FN | 准确率分/75 |
|---|---|---:|---|---|---:|
| 当前冻结流水线复跑 | [46.2, 45.9, 45.5] | 45.9 | 45.5–46.2 | 86/1/6 | 71.3854 |
| 单线程OpenCV＋同卡模型分支并发 | [46.8, 47.7, 46.3] | 46.8 | 46.3–47.7 | 86/1/6 | 71.3854 |

候选相对同期基准中位数变化：+1.96%。全部候选中单次最高：optimized_repeat_2，47.7 FPS。单次最高不是可重复速度保证。

按当前候选中位数套用题目公式，速度分约23.4327/25，合计约94.8181/100。这是15段公开视频的本机复现，不是官方隐藏测试榜单成绩。

## 对照口径与准确性

- 真实单张RTX4090，CUDA_VISIBLE_DEVICES=0，15段视频、2109帧，每轮官方run.py，decoder=gpu，GPU任务顺序执行。
- 25轮完整模型评测全部86 TP、1 FP、6 FN；原86个GT命中身份全部保留，没有新增误检事件。
- 所有预测文件与原NVDEC冻结结果逐字节一致：True。SHA256为7be60833053d1bb69765186eb060b350e69d75907e408440a2ad0db46a022afb。
- 每轮15/15成功，输出schema零错误，无CPU回退。每轮均采到正NVDEC利用率：True；GPU显存采样最大1418 MiB。
- 各轮使用同一组冻结TensorRT缓存，权重、精度、阈值、图像尺寸、球台逐帧推理和事件规则均相同。全部异步任务在finish_video内等待结束，计入官方infer_sec。
- 此处速度是热引擎的官方infer_fps，准备耗时另列；本轮没有重新构建引擎，不将历史冷启动时间冒充本轮测量。
- CPU、驱动、数据和系统负载也会影响FPS；结果不代表所有单卡4090机器的上限。

## 瓶颈证据

无模型、逐帧读取官方GPU解码输出的两次全量诊断分别是47.1、45.2 FPS。这与完整模型方案接近，说明当前环境的解码/读取路径是主要限制之一。这不是4090硬件极限的证明。

提前预处理的pre2运行中，主线程读取解码帧累计38.71秒，process_frame及finish_video累计3.30秒；异步计算与读取重叠，不能把这两项当作GPU纯计算时间。

将本进程拥有的FFmpeg管道由64 KiB扩大到1 MiB，纯解码44.6 FPS，没有收益。它只改该次运行的管道容量，没有改全局sysctl或官方代码，也没有进入默认方案。

worker.pstats只覆盖首段125帧，定位到缩放和归一化热点；dump_stats之后停止profiling，不作为2109帧完整profile证据。

## 逐项实验

| 运行 | 全量FPS | 推理秒 | 准备秒 | TP/FP/FN | 预测完全一致 |
|---|---:|---:|---:|---|---|
| affinity16 | 43.4 | 48.6 | 0.8 | 86/1/6 | True |
| affinity8 | 43.7 | 48.3 | 0.7 | 86/1/6 | True |
| baseline_1 | 46.2 | 45.6 | 0.8 | 86/1/6 | True |
| baseline_repeat_1 | 46.2 | 45.7 | 0.7 | 86/1/6 | True |
| baseline_repeat_2 | 45.9 | 46.0 | 0.8 | 86/1/6 | True |
| baseline_repeat_3 | 45.5 | 46.3 | 0.8 | 86/1/6 | True |
| cv1 | 46.5 | 45.4 | 0.8 | 86/1/6 | True |
| cv1_concurrent | 47.5 | 44.4 | 0.9 | 86/1/6 | True |
| cv1_norm | 46.0 | 45.9 | 0.8 | 86/1/6 | True |
| cv1_norm_concurrent | 46.3 | 45.5 | 0.9 | 86/1/6 | True |
| cv4 | 45.7 | 46.1 | 0.8 | 86/1/6 | True |
| cv8 | 45.4 | 46.5 | 0.8 | 86/1/6 | True |
| norm | 47.1 | 44.7 | 0.9 | 86/1/6 | True |
| optimized_repeat_1 | 46.8 | 45.0 | 0.9 | 86/1/6 | True |
| optimized_repeat_2 | 47.7 | 44.2 | 0.9 | 86/1/6 | True |
| optimized_repeat_3 | 46.3 | 45.5 | 0.8 | 86/1/6 | True |
| owned | 46.9 | 45.0 | 0.8 | 86/1/6 | True |
| pipe_pipeline | 45.2 | 46.7 | 0.8 | 86/1/6 | True |
| pipe_preprocess | 47.0 | 44.9 | 0.8 | 86/1/6 | True |
| pre1 | 46.5 | 45.4 | 0.8 | 86/1/6 | True |
| pre2 | 46.0 | 45.8 | 0.8 | 86/1/6 | True |
| pre4 | 47.1 | 44.8 | 0.8 | 86/1/6 | True |
| profile | 45.5 | 46.3 | 0.8 | 86/1/6 | True |
| queue8 | 47.1 | 44.8 | 0.8 | 86/1/6 | True |
| torch1 | 46.5 | 45.3 | 0.7 | 86/1/6 | True |

实验覆盖OpenCV/PyTorch线程数、CHW原地等价归一化、同卡分支并发、帧缓冲复制与队列深度、CPU亲和性、1/2/4线程提前预处理、局部管道容量。没有依据GT或视频ID选择不同策略。

## 代码与复现

默认participant.solution:Solution仍指向原流水线；本轮并发候选为participant.speed_candidate:Solution。原始freeze目录及权重没有修改。parallel_solution在独立副本中仅新增默认关闭的parallel_on_single_gpu开关。

```bash
cd /root/autodl-tmp/pingpang_4090_20261018/speed50_20261018
bash run_one.sh new_unique_baseline participant.parallel_solution:Solution
bash run_one.sh new_unique_candidate participant.speed_candidate:Solution
```

run_one.sh为本机实验启动器：只暴露GPU0，使用之前已验证的进程级NVDEC兼容库，复制固定引擎缓存，运行完整性校验、官方run.py和冻结评分器。实际参赛模块只带ONNX；服务器兼容库没有打入参赛participant代码。

实验候选仍能识别可见GPU数量，最多使用两张卡，但本轮仅测试一张真实4090，不声明双卡加速或双卡正确性已实测。

pipeline_tests.json验证预处理逐像素一致、球台后处理一致、队列顺序/界限/重置/缓冲所有权、异常传播及临时方法恢复；execution_tests.json验证并发候选的队列契约和0/1/2/8卡选择逻辑。

## 冻结与证据保存

原precision_rescue的997文件及nvdec_parallel的986文件全部复核未变，见freeze_before.json和freeze_after.json。每轮官方完整性校验通过。

本目录FREEZE.json覆盖可下载的代码、报告、结果、日志与一套reference_engines。重复trt_cache仍留在服务器，每份缓存的哈希记录在engine_hashes.json；下载包不重复存放相同缓存。后续复现应写入新的实验目录，以保持本目录冻结。

下一步若继续追求稳定50 FPS，应优先在比赛同款环境核对官方解码吞吐，以及驱动、CPU资源和RGB传输耗时。当前没有证据支持继续调整检测阈值能够解决这个速度瓶颈。
