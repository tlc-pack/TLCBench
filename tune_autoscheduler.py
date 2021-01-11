import os
import argparse

import tvm
from tvm import relay, auto_scheduler

from utils import get_network, make_network_key


network_to_n_trials = {
    # CPU
    ("resnet_50", 1, "float32", "llvm"): 22000,
    ("mobilenet_v2", 1, "float32", "llvm"): 16000,
    ("bert", 1, "float32", "llvm"): 12000,
    # GPU
    ("resnet_50", 1, "float32", "cuda"): 20000,
    ("mobilenet_v2", 1, "float32", "cuda"): 16000,
    ("bert", 1, "float32", "cuda"): 10000,
}


def auto_scheduler_tune(network, batch_size, dtype, target, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)

    layout = "NHWC"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout
    )

    n_trials = network_to_n_trials[(network, batch_size, dtype, str(target.kind))]

    if "cpu" in target.keys:
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=300, timeout=10
        )
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet_50", "mobilenet_v2", "bert", "all"],
        default="all",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=platinum-8124m -mcpu=skylake-avx512",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    parser.add_argument(
        "--logdir", type=str, default="tmp_logs/", help="Log file directory."
    )
    args = parser.parse_args()

    if args.network == "all":
        networks = ["resnet_50", "mobilenet_v2", "bert"]
    else:
        networks = [args.network]
    batch_sizes = [args.batch_size]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    for network in networks:
        for batch_size in batch_sizes:
            for dtype in dtypes:
                network_key = make_network_key(network, batch_size, dtype)
                print("Tune %s ..." % network_key)

                log_file = os.path.join(
                    args.logdir, "autoscheduler", target.model, network_key + ".json"
                )

                auto_scheduler_tune(network, batch_size, dtype, target, log_file)
