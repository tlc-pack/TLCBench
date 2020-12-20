import os
import argparse

import tvm
from tvm import relay, auto_scheduler
from util import get_network


def auto_scheduler_tuning_opt(log_file, dtype = "float32"):
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400, timeout=10)
    return auto_scheduler.TuningOptions(
        num_measure_trials=50,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        runner=measure_ctx.runner,
        verbose=2,
    )

def auto_scheduler_tune(network, target, input_name, log_file):
    if os.path.exists(log_file):
        os.remove(log_file)
    mod, net_params, input_shape, output_shape = get_network(network)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], net_params, target)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(auto_scheduler_tuning_opt(log_file))


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

    for network in networks:
        log_file = os.path.join(args.logdir, "autoscheduler_" + str(target)[:4] + "_" + network + ".log")
        auto_scheduler_tune(network, target, args.inputname, log_file)
