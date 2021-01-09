import os
import argparse

import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner

from utils import get_network, make_network_key, use_graph_tuner


def autotvm_tune(network, batch_size, dtype, target, log_prefix):
    kernel_log = log_prefix + ".kernel.log"
    graph_log = log_prefix + ".graph.log"
    os.makedirs(os.path.dirname(graph_log), exist_ok=True)
    if os.path.exists(kernel_log):
        os.remove(kernel_log)
    if os.path.exists(graph_log):
        os.remove(graph_log)

    layout = "NCHW"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout
    )
    tuning_opt = get_tuning_option(target, kernel_log)
    ops = [
        relay.op.get("nn.batch_matmul"),
        relay.op.get("nn.dense"),
        relay.op.get("nn.conv2d"),
    ]

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=ops
    )
    tune_kernels(tasks, **tuning_opt)

    if use_graph_tuner(network, batch_size, dtype, target):
        tune_graph(mod["main"], input_name, input_shape, target, kernel_log, graph_log)


def get_tuning_option(target, log_file, dtype="float32"):
    if "cpu" in target.keys:
        tuning_option = {
            "log_filename": log_file,
            "tuner": "random",
            "n_trial": 1300,
            "early_stopping": None,
            "use_transfer_learning": False,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                # runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
                runner=autotvm.LocalRunner(
                    number=1, repeat=10, enable_cpu_cache_flush=True
                ),
            ),
        }
    else:
        tuning_option = {
            "log_filename": log_file,
            "tuner": "xgb",
            "n_trial": 2000,
            "early_stopping": 600,
            "use_transfer_learning": True,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(
                    number=20, repeat=3, timeout=4, min_repeat_ms=150
                ),
            ),
        }

    return tuning_option


def tune_kernels(
    tasks,
    measure_option,
    tuner,
    n_trial,
    early_stopping,
    log_filename,
    use_transfer_learning,
):
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(log_filename):
                tuner_obj.load_history(autotvm.record.load_from_file(log_filename))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(
    graph, input_name, input_shape, target, kernel_log, graph_log, use_DP=True
):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: input_shape}, kernel_log, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(graph_log)


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
        "--logdir", type=str, default="tuning_logs/", help="Log file directory."
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

                log_prefix = os.path.join(
                    args.logdir, "autotvm", target.model, network_key
                )
                autotvm_tune(network, batch_size, dtype, target, log_prefix)
