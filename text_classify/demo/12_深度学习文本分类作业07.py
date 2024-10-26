# -*- coding: utf-8 -*-
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn import metrics

print(sys.path)

# 这里可以直接以下列代码import Trainer的主要原因是：sys.path环境变量的文件夹中有text_classify
from text_classify.loss import get_loss_fn
from text_classify.optim import get_train_optim
from text_classify.task.task2.dataset import fetch_dataloader
from text_classify.task.task4.network import NetworkV1, NetworkV2
from text_classify.trainer.base import Trainer

logging.basicConfig(level=logging.INFO)
FILE_ROOT_DIR = os.path.dirname(__file__)


def training():
    # 初始化
    # now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    now_str = "20240607202638"
    # 分字数据的输入
    train_dataloader, test_dataloader, token_2_idx, num_classes, class_weight = fetch_dataloader(
        os.path.join(FILE_ROOT_DIR, "../datas"),
        32
    )
    os.makedirs(f'{FILE_ROOT_DIR}/output/{now_str}', exist_ok=True)
    with open(os.path.join(f'{FILE_ROOT_DIR}/output/{now_str}', "token_2_idx.json"), 'w', encoding="utf-8") as writer:
        json.dump(token_2_idx, writer, ensure_ascii=False, indent=2)
    with open(os.path.join(f'{FILE_ROOT_DIR}/output/{now_str}', "config.json"), 'w', encoding="utf-8") as writer:
        json.dump({
            'num_classes': num_classes,
            'token_numbers': len(token_2_idx)
        }, writer, ensure_ascii=False, indent=2)
    net = NetworkV2(
        vocab_size=len(token_2_idx),
        hidden_size=128,
        num_classes=num_classes
    )
    # NOTE: 能不能每个epoch执行完后，根据评估数据集的效果，来调整各个类别的损失权重值
    loss_fn = get_loss_fn(
        weight=torch.tensor(class_weight, dtype=torch.float32).view(-1),
        label_smoothing=0.1
    )
    train_opt = get_train_optim(
        net, 0.01,
        name='sgd', momentum=0.1, nesterov=True,
        weight_decay=0.002
    )

    # lr_scheduler = get_optim_lr_scheduler(train_opt, name="linear")
    # lr_scheduler = get_optim_lr_scheduler(train_opt, name=lambda epoch: 0.9 ** epoch)
    # lr_scheduler = get_optim_lr_scheduler(train_opt, name=_lr_update_func)

    trainer = Trainer(
        net, loss_fn, train_opt,
        train_dataloader, test_dataloader,
        total_epoch=100,
        summary_log_dir=f'{FILE_ROOT_DIR}/output/{now_str}/summary',
        save_dir=f'{FILE_ROOT_DIR}/output/{now_str}/model',
        example_inputs=torch.randint(100, size=(2, 5)),
        best_model_metric_func=lambda y_true, y_pred: metrics.accuracy_score(y_true, y_pred),
        stop_early=True  # 提前终止模型训练
    )

    # 训练
    trainer.fit()

    # NOTE: 能不能修改代码，让其支持以下功能:
    # 当训练到收敛后，模型持久化保存，然后更改模型的训练参数(eg: 学习率调小、批次增大....)，恢复模型参数继续训练一定的epoch


def export_static_model():
    """
    将训练好的模型转换为静态结构(方便部署)
    可视化:
        https://netron.app/
    :return:
    """
    root_dir = f'{FILE_ROOT_DIR}/output/20240607202638'

    cfg_path = os.path.join(root_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as reader:
        config = json.load(reader)

    net = NetworkV2(
        vocab_size=config['token_numbers'],
        hidden_size=128,
        num_classes=config['num_classes']
    )
    best_param = torch.load(
        os.path.join(root_dir, "model", "best.pkl"),
        map_location="cpu"
    )['param']
    missing_keys, unexpected_keys = net.load_state_dict(best_param, strict=False)
    print(f"未进行参数恢复的参数列表为:{missing_keys}")
    print(f"额外给定的参数列表为:{unexpected_keys}")
    net.eval()

    example_x = torch.tensor([
        [5, 6, 2, 3, 5, 0, 0, 0],
        [1, 2, 5, 3, 6, 8, 5, 3]
    ], dtype=torch.int64)
    example_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=torch.float32)
    example = (example_x, )

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
    root_dir = f'{FILE_ROOT_DIR}/output/20240607202638'

    example_x = torch.tensor([
        [5, 6, 2, 3, 5, 0, 0],
        [5, 6, 2, 3, 5, 0, 0],
        [1, 2, 5, 3, 6, 8, 3]
    ], dtype=torch.int64)
    example_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1]
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
            ['336'],
            input_feed={
                "input.1": example_x.numpy(),
            })[0]
    else:
        onnx_v1_output = None

    onnx_v2_output = onnx_model_v2.run(
        ['output'],
        input_feed={
            "input_x": example_x.numpy(),
            # "input_mask": example_mask.numpy()
        })[0]
    jit_output = jit_model((example_x, )).numpy()

    print("=" * 50)
    print(jit_output)
    print(onnx_v1_output)
    print(onnx_v2_output)
    print(np.mean(np.abs(jit_output - onnx_v2_output) > 1e-5))
    if onnx_v1_output is not None:
        print(np.mean(np.abs(onnx_v2_output - onnx_v1_output) > 1e-8))


if __name__ == '__main__':
    # training()
    # export_static_model()
    load_static_model()
