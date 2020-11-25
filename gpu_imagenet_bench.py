import os
import argparse
import threading

import numpy as np

import tvm
from tvm import te, autotvm, auto_scheduler
import tvm.contrib.graph_runtime as runtime
from tvm import relay

from util import get_network, autotvm_tune, auto_scheduler_tune



def benchmark(network, target, input_name, kernel_log, graph_log, tune=True, method="autotvm"):
    mod, params, input_shape, output_shape = get_network(network, batch_size=1)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    # Tune
    #if not tune and not os.path.exists(graph_log):
    #    # TODO
    #    raise IOError("Pre-tuned file unaccessable")

    #if tune and os.path.exists(graph_log):
    #    os.remove(graph_log)
    
    if method == "autotvm":
        if tune:
            print("Tune...")
            print(graph_log)
            autotvm_tune(network, target, input_name, kernel_log, graph_log)
        print("Compile...")
        with  autotvm.apply_history_best(graph_log):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)
    elif method == "ansor":
        if tune:
            print("Tune...")
            auto_scheduler_tune(network, target, input_name, kernel_log, graph_log)
        print("Compile...")
        with auto_scheduler.ApplyHistoryBest(graph_log):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)
    else:
        raise ValueError("Unsupported tuning method")

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
    parser.add_argument("--repeat", type=int, default=600)
    parser.add_argument(
        "--target",
        type=str,
        choices=["cuda", "opencl", "rocm", "nvptx", "metal", "vulkan"],
        default="cuda",
        help="The tvm compilation target",
    )
    parser.add_argument("--thread", type=int, default=1, help="The number of threads to be run.")
    args = parser.parse_args()

    dtype = "float32"
    input_name = "data"

    if args.network is None:
        networks = ["resnet-50", "mobilenet"]
    else:
        networks = [args.network]

    #target = tvm.target.Target("%s -device=%s -model=%s" % (args.target, args.device, args.model))
    target = tvm.target.cuda()

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    for network in networks:
        kernel_log = "%s_kernel.log" % network
        graph_log = "%s_graph.log" % network
        if args.thread == 1:
            benchmark(network, target, input_name, kernel_log, graph_log)
        else:
            threads = list()
            for n in range(args.thread):
                thread = threading.Thread(
                    target=benchmark, args=([network, target, input_name, kernel_log, graph_log]), name="thread%d" % n
                )
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
