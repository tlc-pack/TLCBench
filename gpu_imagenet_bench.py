import os
import argparse
import threading

import numpy as np

import tvm
from tvm import te, autotvm, auto_scheduler
import tvm.contrib.graph_runtime as runtime
from tvm import relay

from util import get_network, autotvm_tune, auto_scheduler_tune

def benchmark(network, target, log_file):
    mod, params, input_shape, output_shape = get_network(network, batch_size=1)
    if not args.notune:
        print("Tune...")
        if args.tunemethod == "autotvm":
            autotvm_tune(network, target, args.inputname, log_file)
        elif args.tunemethod == "autoscheduler":
            auto_scheduler_tune(network, target, args.inputname, log_file)
        else:
            raise ValueError("Unsupported tuning method")

    print("Compile...")
    if args.tunemethod == "autotvm":
        with  autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)
    
    elif args.tunemethod == "autoscheduler":
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)

    # create runtime
    ctx = tvm.context(str(target), 0)
    module = runtime.GraphModule(lib["default"](ctx))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)

    # evaluate
    print("Evaluate...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=args.repeat)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print(
        "%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=[
            "resnet-50",
            "mobilenet",
        ],
        help="The name of neural network",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["amd_apu"],
        default="amd_apu",
        help="The name of the test device. If your device is not listed in "
        "the choices list, pick the most similar one as argument.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["1080ti", "titanx", "tx2", "gfx900", "v1000"],
        default="1080ti",
        help="The model of the test device. If your device is not listed in "
        "the choices list, pick the most similar one as argument.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["cuda", "opencl", "rocm", "nvptx", "metal", "vulkan"],
        default="cuda",
        help="The tvm compilation target",
    )

    parser.add_argument("--inputname", type=str, default="data", help="Input name of the graph. For ONNX models, it is typically 0")
    parser.add_argument("--repeat", type=int, default=600)
    parser.add_argument("--thread", type=int, default=1, help="The number of threads to be run.")
    parser.add_argument("--logdir", type=str, default="log/", help="Log file directory.")
    parser.add_argument("--notune", type=bool, const=True, default=False, help="Disable tuning.")
    parser.add_argument(
        "--tunemethod", 
        type=str,
        choices=["autotvm", "autoschduler"],
        default="autoscheduler",
        help="Tuning method",
    )

    args = parser.parse_args()
    dtype = "float32"

    if args.network is None:
        networks = ["resnet-50", "mobilenet"]
    else:
        networks = [args.network]

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    target = tvm.target.Target("%s -device=%s -model=%s" % (args.target, args.device, args.model))

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    for network in networks:
        log_file = os.path.join(args.logdir, network + ".log")
        if args.thread == 1:
            benchmark(network, target, log_file)
        else:
            threads = list()
            for n in range(args.thread):
                thread = threading.Thread(
                    target=benchmark, args=([network, target, log_file]), name="thread%d" % n
                )
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
