import os
import argparse

import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_runtime as runtime

from utils import get_network, make_network_key, use_graph_tuner


def benchmark(network, batch_size, dtype, target, log_prefix, repeat):
    layout = "NCHW"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout
    )

    if use_graph_tuner(network, batch_size, dtype, target):
        log_file = log_prefix + ".graph.log"
        history_best_context = autotvm.apply_graph_best(log_file)
    else:
        log_file = log_prefix + ".kernel.log"
        history_best_context = autotvm.apply_history_best(log_file)

    assert os.path.exists(log_file), "The log file '%s' does not exist." % log_file
    print("Use log file %s" % log_file)

    if network in ["bert"]:
        # Build module
        with history_best_context:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
        ctx = tvm.context(str(target), 0)
        module = runtime.GraphModule(lib["default"](ctx))

        # Feed input data
        seq_length = input_shape[0][1]
        data = np.random.uniform(size=input_shape[0])
        token_types = np.random.uniform(size=input_shape[1])
        valid_length = np.array([seq_length] * batch_size)
        module.set_input(data0=data, data1=token_types, data2=valid_length)
    else:
        # Build module
        with history_best_context:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)
        ctx = tvm.context(str(target), 0)
        module = runtime.GraphModule(lib["default"](ctx))

        # Feed input data
        data = np.random.uniform(size=input_shape)
        module.set_input(input_name, data)

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=repeat)
    return np.array(ftimer().results)


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
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.network == "all":
        networks = ["resnet_50", "mobilenet_v2", "bert"]
    else:
        networks = [args.network]
    batch_sizes = [args.batch_size]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    # Benchmark
    result_messages = []
    for network in networks:
        for batch_size in batch_sizes:
            for dtype in dtypes:
                network_key = make_network_key(network, batch_size, dtype)
                print("Benchmark %s ..." % network_key)

                log_prefix = os.path.join(
                    args.logdir, "autotvm", target.model, network_key
                )
                prof_res = benchmark(
                    network, batch_size, dtype, target, log_prefix, args.repeat
                )

                prof_res *= 1000  # convert to millisecond
                message = "%-18s %-12s %-19s (%s)" % (
                    network,
                    batch_size,
                    "%.2f ms" % np.mean(prof_res),
                    "%.2f ms" % np.std(prof_res),
                )
                result_messages.append(message)

    # Print result
    print("-------------------------------------------------------------")
    print(
        "%-18s %-12s %-20s"
        % ("Network Name", "Batch size", "Mean Inference Time (std dev)")
    )
    print("-------------------------------------------------------------")
    for line in result_messages:
        print(line)
    print("-------------------------------------------------------------")
