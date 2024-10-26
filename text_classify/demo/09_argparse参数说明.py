# -*- coding: utf-8 -*-
import argparse


def training():
    print("执行模型训练的方法")


def eval():
    print("执行模型评估的方法")


# 参数的定义
# 参考页面: https://docs.python.org/zh-cn/dev/library/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', default="train", type=str, choices=['train', 'eval'], help="给定任务名称")
parser.add_argument('-e', '--epochs', default=100, type=str, help="给定迭代的epoch总次数")

args = parser.parse_args()
print(args)

if args.task == 'train':
    training()
elif args.task == 'eval':
    eval()
else:
    print(f"当前任务参数异常:{args.task}")

print("你好")
