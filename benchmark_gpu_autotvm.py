import os
import argparse
import threading

import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_runtime as runtime

from util import get_network


def benchmark(network, target, log_file):
    mod, params, input_shape, output_shape = get_network(network)
    # covert to NCHW
    desired_layouts = {"nn.conv2d": ["NCHW", "default"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    if network in ["bert"]:
        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)
                # upload parameters to device
            ctx = tvm.context(str(target), 0)
            data_tvm = tvm.nd.array(
                (np.random.uniform(size=input_shape[0])).astype(dtype)
            )
            token_types_tvm = tvm.nd.array(
                np.random.uniform(size=input_shape[1]).astype(dtype)
            )
            valid_length_tvm = tvm.nd.array(
                np.random.uniform(size=input_shape[2]).astype(dtype)
            )
            module = runtime.GraphModule(lib["default"](ctx))
            module.set_input(
                data0=data_tvm, data1=token_types_tvm, data2=valid_length_tvm
            )
    else:
        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)

            # upload parameters to device
            ctx = tvm.context(str(target), 0)
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
            module = runtime.GraphModule(lib["default"](ctx))
            module.set_input(args.inputname, data_tvm)

    # evaluate
    print("Evaluate...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=args.repeat)
    prof_res = (
        np.array(ftimer().results) * 1000
    )  # multiply 1000 for converting to millisecond
    print(
        "%-20s %-19s (%s)"
        % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet-50", "mobilenet_v2", "bert", "all"],
        help="The name of neural network",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=e5-2670 -mcpu=core-avx2",
        help="The tvm compilation target",
    )
    parser.add_argument(
        "--logdir", type=str, default="tuning_logs/", help="Log file directory."
    )
    parser.add_argument(
        "--thread", type=int, default=1, help="The number of threads to be run."
    )
    parser.add_argument(
        "--inputname",
        type=str,
        default="data",
        help="Input name of the graph. For ONNX models, it is typically 0",
    )
    parser.add_argument("--repeat", type=int, default=100)

    args = parser.parse_args()
    dtype = "float32"

    if args.network is None or args.network == "all":
        networks = ["resnet-50", "mobilenet_v2", "bert"]
    else:
        networks = [args.network]

    if not os.path.exists(args.logdir):
        print("[Error] Tuning log dir %s does not exist" % args.logdir)

    target = tvm.target.Target(args.target)

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")

    target_name = "gpu"
    for network in networks:
        log_file = os.path.join(
            args.logdir, "autotvm_" + target_name + "_" + network + "_kernel.log"
        )
        if args.thread == 1:
            benchmark(network, target, log_file)
        else:
            threads = list()
            for n in range(args.thread):
                thread = threading.Thread(
                    target=benchmark,
                    args=([network, target, log_file]),
                    name="thread%d" % n,
                )
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
