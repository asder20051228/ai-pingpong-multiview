"""Only replace ball inference; frozen rules/table/decoding remain identical."""
from pathlib import Path
import time
from .final_solution import Solution as FrozenColor
ROOT=Path(__file__).resolve().parent
def strict_model(context):
    import tensorrt as trt
    from core.trt import TrtModel
    onnx=ROOT/'frozen85/participant/weights/ball.onnx'
    import torch, hashlib
    gpu=torch.cuda.get_device_name(torch.cuda.current_device()).replace('/', '_')
    key=hashlib.sha256(onnx.read_bytes()).hexdigest()[:16]
    engine=Path(context.output_dir)/'trt_cache'/trt.__version__/gpu/('strict_no_tf32_'+key+'.engine')
    engine.parent.mkdir(parents=True,exist_ok=True)
    if not engine.exists():
        start=time.monotonic();logger=trt.Logger(trt.Logger.WARNING)
        builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(network,logger);config=builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
        config.clear_flag(trt.BuilderFlag.FP16);config.clear_flag(trt.BuilderFlag.TF32)
        shape=(1,9,288,512);profile=builder.create_optimization_profile();profile.set_shape('input',shape,shape,shape);config.add_optimization_profile(profile)
        assert parser.parse(onnx.read_bytes()),[str(parser.get_error(i)) for i in range(parser.num_errors)]
        print('BUILD_STRICT_FP32',flush=True)
        serialized=builder.build_serialized_network(network,config);assert serialized
        engine.write_bytes(serialized)
        import core.trt as stats
        stats.BUILD_SECONDS_TOTAL += time.monotonic()-start
        print('BUILT_STRICT_FP32_SECONDS',time.monotonic()-start,flush=True)
    return TrtModel(engine)
class Solution(FrozenColor):
    def prepare(self,context):
        super().prepare(context)
        self.strict=strict_model(context)
        def infer(x):
            y=self.strict.infer(x);self.latest_heat=y[0];return y
        self.det._infer=infer
        import numpy as np
        infer(np.zeros((1,9,288,512),np.float32))
        context.info['backend']='strict_fp32_ball_no_tf32_table_fp16'
