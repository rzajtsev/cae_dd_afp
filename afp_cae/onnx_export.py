"""Conversion of the trained JAX parameters into an ONNX graph (opset 17, NCHW layout).

The graph mirrors `afp_cae.model.forward` exactly; the weights are transposed from the JAX
layouts (HWIO convolutions, NHWC flattening) to the ONNX ones (OIHW, NCHW flattening).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .model import BOTTOM, ENC, NEG_SLOPE, TILE


def build_onnx(params) -> onnx.ModelProto:
    p = {k: v for k, v in params.items()}
    nodes, inits = [], []
    c_bot = ENC[-1][1]
    flat = BOTTOM * BOTTOM * c_bot

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def conv(x, layer, name, stride):
        w = const(name + "_w", np.transpose(np.asarray(layer["w"]), (3, 2, 0, 1)).astype(np.float32))  # HWIO->OIHW
        b = const(name + "_b", np.asarray(layer["b"], np.float32))
        nodes.append(helper.make_node("Conv", [x, w, b], [name], kernel_shape=[3, 3],
                                      strides=[stride, stride], auto_pad="SAME_UPPER"))
        return name

    def lrelu(x, name):
        nodes.append(helper.make_node("LeakyRelu", [x], [name], alpha=NEG_SLOPE))
        return name

    x = "x"
    for i, layer in enumerate(p["enc"]):
        x = lrelu(conv(x, layer, f"e{i}", 2), f"e{i}a")
    # JAX flattens NHWC as (h, w, c); ONNX Flatten on NCHW gives (c, h, w): permute FC rows
    we = np.asarray(p["fc_e"]["w"]).reshape(BOTTOM, BOTTOM, c_bot, -1).transpose(2, 0, 1, 3).reshape(flat, -1)
    nodes.append(helper.make_node("Flatten", [x], ["flat"], axis=1))
    nodes.append(helper.make_node("Gemm", ["flat", const("fce_w", we.astype(np.float32)),
                                           const("fce_b", np.asarray(p["fc_e"]["b"], np.float32))], ["z"]))
    wd = np.asarray(p["fc_d"]["w"]).reshape(-1, BOTTOM, BOTTOM, c_bot).transpose(0, 3, 1, 2).reshape(-1, flat)
    bd = np.asarray(p["fc_d"]["b"]).reshape(BOTTOM, BOTTOM, c_bot).transpose(2, 0, 1).reshape(-1)
    nodes.append(helper.make_node("Gemm", ["z", const("fcd_w", wd.astype(np.float32)),
                                           const("fcd_b", bd.astype(np.float32))], ["d0"]))
    lrelu("d0", "d0a")
    nodes.append(helper.make_node("Reshape", ["d0a", const("shape", np.array([-1, c_bot, BOTTOM, BOTTOM], np.int64))],
                                  ["d0r"]))
    x = "d0r"
    scales = const("scales", np.array([1, 1, 2, 2], np.float32))
    for i, layer in enumerate(p["dec"]):
        nodes.append(helper.make_node("Resize", [x, "", scales], [f"u{i}"], mode="nearest"))
        x = lrelu(conv(f"u{i}", layer, f"d{i + 1}", 1), f"d{i + 1}a")
    x = conv(x, p["out"], "o", 1)
    nodes.append(helper.make_node("Sigmoid", [x], ["xhat"]))
    graph = helper.make_graph(
        nodes, "afp_cae",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, TILE, TILE])],
        [helper.make_tensor_value_info("xhat", TensorProto.FLOAT, ["N", 1, TILE, TILE])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    model.doc_string = "Convolutional autoencoder for AFP defect detection (input/output: N x 1 x 64 x 64, [0, 1])"
    onnx.checker.check_model(model)
    return model
