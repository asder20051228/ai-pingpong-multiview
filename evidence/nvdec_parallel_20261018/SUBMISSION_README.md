# 自动单卡/双卡参赛代码

正式入口：`participant.solution:Solution`。

将 `submission/pingpang/participant` 目录复制到参赛工程的乒乓球任务中；保持官方 `core/`、`run.py`、`validate.py`、`run_all.sh` 和篮球任务原样。这里的 `submission/pingpang` 也保留了一份通过完整性校验的官方框架，便于独立复现。

```bash
cd submission/pingpang
python integrity.py --verify
python run.py --input /path/to/pingpang_input --output /path/to/new_output \
  --solution participant.solution:Solution --decoder gpu --timeout-sec 7200
```

## 执行方式

- 自动读取 `torch.cuda.device_count()`，遵守启动前设置的 `CUDA_VISIBLE_DEVICES`。
- 1张卡：GPU0运行两个模型，CPU解码与推理线程重叠。
- 2张及以上可见卡：最多使用逻辑GPU0、GPU1；球模型与FP32复核在GPU0，球台模型在GPU1，并发完成后按原帧序更新轨迹。
- 0张卡：明确报错；TensorRT版不伪装成已使用GPU。
- 每次最多2帧排队，视频结束前等待全部工作。输出与计时包括所有帧处理。
- `run_status.json/solution_info/gpu_execution` 记录实际可见卡数、分配与并行方式。
- `parallel_solution.SameGpuBranchDiagnostic` 仅用于同卡并发诊断，正式入口不使用它。

## 权重和缓存

所有权重在 `participant/frozen85/participant/weights`，参赛包只带ONNX。首次运行现场构建TensorRT引擎，写入输出目录的 `trt_cache`；热缓存可复用。严格FP32复核模型禁用TF32，按TensorRT版本、GPU名称、ONNX内容哈希分缓存。

已测试：Python3.10、PyTorch2.1.2+cu118、TensorRT8.6.1、单张4090。当前真实双卡尚未验证；仅完成卡数选择单元测试与同卡双线程全量对照。

## NVDEC

这份参赛代码不改官方解码器。当前租用服务器的580.105.08驱动存在部分GPU分配时NVDEC枚举失败问题；本次在服务器通过独立的 `with_nvdec.sh` 进程兼容启动器完成真实NVDEC测试。

兼容启动器和GPL来源库保存在实验根目录，没有放进此参赛模块。正常NVDEC环境直接使用上述正式命令。若比赛平台也遇到同类宿主驱动问题，需要平台处理或确认其环境兼容措施许可，不能将本地环境处理默认为组委会批准。

## 保留基准

原86命中版本与其997文件冻结清单均保留，未被新代码覆盖。新代码的完整全量结果、速度范围、冷启动耗时见实验根目录REPORT.md。
