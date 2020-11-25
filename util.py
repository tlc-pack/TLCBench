import os

from tvm import relay, autotvm, auto_scheduler
from tvm.relay.testing import resnet
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner

def get_network(name, batch_size=1, dtype="float32"):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

############################################ Tuning for AutoTVM *********************************
def autotvm_tune(network, target, input_name, log_file):
    mod, params, input_shape, output_shape = get_network(network)
    tasks = autotvm.task.extract_from_program(
            mod["main"], target=target,
            params=params, ops=(relay.op.get("nn.conv2d"),)
    )
    if os.path.exists(log_file):
        os.remove(log_file)

    if target.keys[0] == "cpu":
        tmp_log = "tmp.log"
        tuning_opt = autotvm_tuning_opt(target, network, tmp_log)
        tune_kernels(tasks, **tuning_opt)
        tune_graph(mod["main"], input_shape, tmp_log,
                log_file, target, input_name)
        os.remove(tmp_log)
    else:
        tuning_opt = autotvm_tuning_opt(target, network, log_file)
        tune_kernels(tasks, **tuning_opt)

def autotvm_tuning_opt(target, network, log_file, dtype = "float32"):
    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 1,
        "early_stopping": 1,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }
    return tuning_option

def tune_kernels(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log"
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
def tune_graph(graph, dshape, records, opt_sch_file, target, input_name, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

############################################ Tuning for Auto Schedule *********************************
def auto_scheduler_tuning_opt(network, target, log_file, dtype = "float32"):
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400, timeout=10)
    return auto_scheduler.TuningOptions(
        num_measure_trials=200,
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
    tuner.tune(auto_scheduler_tuning_opt(network, target, log_file))
