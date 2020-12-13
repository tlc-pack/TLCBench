import os

import tvm
from tvm import relay
from tvm.relay.testing import resnet

import mxnet
import gluonnlp
from mxnet.gluon.model_zoo.vision import get_model

def get_network(name, batch_size=1, dtype="float32"):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    seq_length = 128
    multiplier = 0.5 # for mobilenet

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        block = mxnet.gluon.model_zoo.vision.get_resnet(1, n_layer, pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
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
    elif name == "mobilenet_v2":
        block = mxnet.gluon.model_zoo.vision.get_mobilenet_v2(multiplier, pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    elif name == "bert":
        # Instantiate a BERT classifier using GluonNLP
        model_name = 'bert_12_768_12'
        dataset = 'book_corpus_wiki_en_uncased'
        model, _ = gluonnlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False)

        # Convert the MXNet model into TVM Relay format
        input_shape = {
            'data0': (batch_size, seq_length),
            'data1': (batch_size, seq_length),
            'data2': (batch_size,)
        }
        mod, params = relay.frontend.from_mxnet(model, shape_dict)

    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


if __name__ == "__main__":
    print(get_network("resnet-50"))
