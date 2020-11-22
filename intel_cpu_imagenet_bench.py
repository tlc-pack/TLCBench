import os
import os.path
import argparse
import threading

import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_runtime as runtime

from util import autotvm_tune, auto_scheduler_tune, get_network

def tune(network, target, input_name, tune_kernel_log, tune_graph_log, method):
    if method == "autotvm":
        lib = autotvm_tune(network, target, input_name, tune_kernel_log, tune_graph_log)
    elif method == "ansor":
        lib = auto_scheduler_tune(network, target, input_name, tune_kernel_log, tune_graph_log)
    else:
        raise ValueError("Unsupported scheduler: " + name)

def benchmark(network, target, dtype, input_name, tune_kernel_log, tune_graph_log):
    mod, net_params, input_shape, output_shape = get_network(network)
    method = "autotvm"
    # run tuning tasks
    print("Tuning...")
    if (not os.path.exists(tune_graph_log)):
        tune(network, target, input_name, tune_kernel_log, tune_graph_log, method)

    print("Compile...")
    if method == "autotvm":
        # compile kernels with graph-level best records
        with autotvm.apply_graph_best(tune_graph_log):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=net_params)  
    else:
        return   

    # upload parameters to device
    ctx = tvm.context(str(target), 0)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](ctx))
    module.set_input(input_name, data_tvm)

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
        "--mcpu",
        type=str,
        choices=[
            "core-avx2", "skylake-avx512"
        ],
        default="core-avx2",
        help="The name of the test device. If your device is not listed in "
        "the choices list, pick the most similar one as argument.",
    )
    parser.add_argument("--logdir", type=str, default="log", help="log directory")
    parser.add_argument("--inputname", type=str, default="data", help="Input name of the graph. For ONNX models, it is typically 0")
    parser.add_argument("--repeat", type=int, default=600)
    parser.add_argument("--thread", type=int, default=1, help="The number of threads to be run.")
    args = parser.parse_args()
    
    dtype = "float32"
    target = "llvm"

    if args.network is None:
        networks = ["resnet-50", "mobilenet"]
    else:
        networks = [args.network]

    target = tvm.target.Target("%s --mcpu=%s" % (target, args.mcpu))
    if (not os.path.isdir(args.logdir)):
        os.mkdir(args.logdir)
        
    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    for network in networks:
        tune_kernel_log = "%s/%s_tune_kernel.log" % (args.logdir, network)
        tune_graph_log = "%s/%s_tune_graph.log" % (args.logdir, network)

        if args.thread == 1:
            benchmark(network, target, dtype, args.inputname, 
                        tune_kernel_log, tune_graph_log)
        else:
            threads = list()
            for n in range(args.thread):
                thread = threading.Thread(
                    target=benchmark, args=([network, target, dtype, args.inputname, 
                                            tune_kernel_log, tune_graph_log]), name="thread%d" % n
                )
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()