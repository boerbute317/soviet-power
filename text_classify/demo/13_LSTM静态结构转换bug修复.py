# -*- coding: utf-8 -*-
import os
# noinspection PyUnresolvedReferences
from datetime import datetime

import numpy as np
import torch

from text_classify.task.task4.network import NetworkV1
from text_classify.task.task5.network import NetworkV3

FILE_ROOT_DIR = os.path.dirname(__file__)


def export_static_model():
    """
    将训练好的模型转换为静态结构(方便部署)
    可视化:
        https://netron.app/
    :return:
    """
    # now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    now_str = "20240612203649"
    root_dir = f'{FILE_ROOT_DIR}/output/{now_str}'
    os.makedirs(os.path.join(root_dir, "model"), exist_ok=True)

    net = NetworkV1(
        vocab_size=10000,
        hidden_size=128,
        num_classes=12
    )
    net.eval()

    example_x = torch.tensor([
        [1, 2, 5, 3, 6, 8, 5, 3],
        [5, 6, 2, 3, 5, 0, 0, 0]
    ], dtype=torch.int64)
    example_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0]
    ], dtype=torch.float32)
    example = (example_x, example_mask)

    # 静态结构的转换

    # TorchScript结构转换：主要用于Torch框架在不同语言中的部署加载
    # https://pytorch.org/docs/stable/jit.html
    # https://pytorch.org/tutorials/advanced/cpp_export.html
    jit_net = torch.jit.trace(net, (example,))
    torch.jit.save(jit_net, os.path.join(root_dir, "model", "best.pt"))

    # ONNX结构
    # https://pytorch.org/docs/stable/onnx.html
    # https://onnxruntime.ai/
    torch.onnx.export(
        net.eval().cpu(),  # 模型对象
        (example,),  # 输入的案例样本tensor对象
        f=os.path.join(root_dir, "model", "best.onnx"),
        input_names=None,
        output_names=None,
        opset_version=12,
        dynamic_axes=None  # None表示固定shape输入，固定的shape就是输入案例的shape大小
    ),
    torch.onnx.export(
        net.eval().cpu(),  # 模型对象
        (example,),  # 输入的案例样本tensor对象
        f=os.path.join(root_dir, "model", "best_dynamic.onnx"),
        input_names=['input_x', 'input_mask'],
        output_names=['output'],
        opset_version=12,
        dynamic_axes={
            'input_x': {
                0: 'n',
                1: 't'
            },
            'input_mask': {
                0: 'n',
                1: 't'
            },
            'output': {
                0: 'n'
            }
        }
    ),

    print("静态结构转换成功！！")


@torch.no_grad()
def load_static_model():
    root_dir = f'{FILE_ROOT_DIR}/output/20240612203649'

    example_x = torch.tensor([
        [1, 2, 5, 3, 6, 8, 3],
        [5, 6, 2, 3, 5, 0, 0],
        [5, 6, 2, 3, 5, 0, 0]
    ], dtype=torch.int64)
    example_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ], dtype=torch.float32)

    # torch script模型加载
    jit_model = torch.jit.load(os.path.join(root_dir, "model", "best.pt"), map_location='cpu')
    jit_model.eval().cpu()

    # ONNX格式的模型加载
    import onnxruntime as ort

    onnx_model_v1 = ort.InferenceSession(os.path.join(root_dir, "model", "best.onnx"))
    onnx_model_v2 = ort.InferenceSession(os.path.join(root_dir, "model", "best_dynamic.onnx"))

    print("=" * 10, "v1")
    print([t.name for t in onnx_model_v1.get_inputs()])
    print([t.name for t in onnx_model_v1.get_outputs()])
    print("=" * 10, "v2")
    print([t.name for t in onnx_model_v2.get_inputs()])
    print([t.name for t in onnx_model_v2.get_outputs()])

    if example_x.shape[0] == 2 and example_x.shape[1] == 8:
        onnx_v1_output = onnx_model_v1.run(
            ['344'],
            input_feed={
                "input.1": example_x.numpy(),
                "onnx::ReduceSum_1": example_mask.numpy()
            })[0]
    else:
        onnx_v1_output = None

    onnx_v2_output = onnx_model_v2.run(
        ['output'],
        input_feed={
            "input_x": example_x.numpy(),
            "input_mask": example_mask.numpy()
        })[0]
    jit_output = jit_model((example_x, example_mask)).numpy()

    print("=" * 50)
    print(jit_output)
    print(onnx_v1_output)
    print(onnx_v2_output)
    print(np.mean(np.abs(jit_output - onnx_v2_output) > 1e-5))
    if onnx_v1_output is not None:
        print(np.mean(np.abs(onnx_v2_output - onnx_v1_output) > 1e-8))


if __name__ == '__main__':
    export_static_model()
    load_static_model()
