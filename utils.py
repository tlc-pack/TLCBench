import os

import tvm
from tvm import relay


def get_network(name, batch_size, dtype, layout):
    """Get the symbol definition and random weight of a network"""
    input_name = "data"
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        import mxnet

        n_layer = int(name.split("_")[1])
        block = mxnet.gluon.model_zoo.vision.get_resnet(1, n_layer, pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    elif name == "mobilenet_v2":
        import mxnet

        multiplier = 1
        block = mxnet.gluon.model_zoo.vision.get_mobilenet_v2(
            multiplier, pretrained=True
        )
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    elif name == "bert":
        import gluonnlp

        seq_length = 128

        # Instantiate a BERT classifier using GluonNLP
        model_name = "bert_12_768_12"
        dataset = "book_corpus_wiki_en_uncased"
        model, _ = gluonnlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False,
        )

        # Convert the MXNet model into TVM Relay format
        shape_dict = {
            "data0": (batch_size, seq_length),
            "data1": (batch_size, seq_length),
            "data2": (batch_size,),
        }
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        input_shape = (shape_dict["data0"], shape_dict["data1"], shape_dict["data2"])

        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(
            lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
                fn, params
            ),
            opt_level=1,
        )
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_name, input_shape, output_shape


def make_network_key(network_name, batch_size, dtype):
    return "%s-B%s-%s" % (network_name, batch_size, dtype)


def use_graph_tuner(network_name, batch_size, dtype, target):
    """Return whether use graph tuner for a network on a target"""
    # Only use graph tuner for CNNs on CPUs
    return "cpu" in target.keys and not (network_name in ["bert"])


def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod
