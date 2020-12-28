import os
import argparse

import tvm
from tvm import relay, auto_scheduler
from util import get_network

def auto_scheduler_tune(network, target, input_name, log_file):
    if os.path.exists(log_file):
        os.remove(log_file)
    mod, net_params, input_shape, output_shape = get_network(network)
    if network not in ["bert"]:
        # convert to NHWC layout
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)

    if "cpu" in target.keys:
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=20000,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=20000,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], net_params, target)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=[
            "resnet-50",
            "mobilenet_v2",
            "bert",
            "all"
        ],
        help="The name of neural network",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=e5-2670 -mcpu=core-avx2",
        help="The tvm compilation target",
    )
    parser.add_argument("--logdir", type=str, default="tuning_logs/", help="Log file directory.")
    parser.add_argument("--thread", type=int, default=1, help="The number of threads to be run.")
    parser.add_argument("--inputname", type=str, default="data",
                        help="Input name of the graph. For ONNX models, it is typically 0")

    args = parser.parse_args()
    dtype = "float32"

    if args.network is None or args.network == "all":
        networks = ["resnet-50", "mobilenet_v2", "bert"]
    else:
        networks = [args.network]

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    target = tvm.target.Target(args.target)

    if "cpu" in target.keys:
        target_name = "cpu"
    else:
        target_name = "cuda"
    for network in networks:
        log_file = os.path.join(args.logdir, "autoscheduler_" + target_name + "_" + network + ".log")
        auto_scheduler_tune(network, target, args.inputname, log_file)
