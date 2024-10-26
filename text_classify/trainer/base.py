# -*- coding: utf-8 -*-
"""
定义一个通用的训练器
"""
import atexit
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self,
                 net, loss_fn, train_opt, train_dataloader, test_dataloader, total_epoch,
                 summary_log_dir, save_dir,
                 example_inputs=None,
                 lr_scheduler=None,
                 best_model_metric_func=None,
                 stop_early=False,
                 stop_early_epoch_interval=5
                 ):
        super(Trainer, self).__init__()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.net: nn.Module = net
        self.loss_fn = loss_fn
        self.train_opt: optim.Optimizer = train_opt
        self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        self.start_epoch = 0

        self.train_step = 0
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(summary_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=summary_log_dir)
        if example_inputs is not None:
            self.writer.add_graph(self.net, example_inputs)
        atexit.register(self.close)  # 在程序结束的时候，主动执行close方法

        self.best_score = float('-inf')
        self.best_model_metric_func = best_model_metric_func
        self.stop_early = (self.best_model_metric_func is not None) and stop_early
        self.is_stop_early = False
        self.stop_early_epoch_interval = stop_early_epoch_interval
        self.stop_early_epoch = 0

        # 模型代码恢复
        model_names = os.listdir(self.save_dir)
        model_path = None
        if 'best.pkl' in model_names:
            model_path = os.path.join(self.save_dir, "best.pkl")
        else:
            model_names = list(filter(lambda name: name.startswith("model_"), model_names))
            model_names = sorted(model_names, key=lambda name: int(name.split(".")[0].split("_")[1]), reverse=True)
            if len(model_names) > 0:
                model_path = os.path.join(self.save_dir, model_names[0])
        if model_path is not None and os.path.exists(model_path):
            print(f"开始进行模型参数恢复:{model_path}")
            save_data = torch.load(model_path, map_location="cpu")
            best_param = save_data['param']
            self.start_epoch = save_data['epoch'] + 1
            self.total_epoch += self.start_epoch
            self.best_score = save_data.get('score', self.best_score)

            # best_param: 实际上就是一个dict字典，key就是参数名称字符串，value就是tensor对象值
            missing_keys, unexpected_keys = net.load_state_dict(best_param, strict=False)
            print(f"未进行参数恢复的参数列表为:{missing_keys}")
            print(f"额外给定的参数列表为:{unexpected_keys}")

    def close(self):
        logging.info("close resources....")
        self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self

    def fit(self):
        logging.info(f"start fit model parameters {self.total_epoch}")
        for epoch in range(self.start_epoch, self.total_epoch):
            logging.info(f"epoch {epoch} / {self.total_epoch}")
            # 训练
            self.train_epoch(epoch)
            # 当一个epoch的所有batch均训练完后，调用学习率更新器更新优化器的学习率
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # 评估
            cur_model_score = self.eval_epoch(epoch)
            # 模型保存
            self.save_epoch(epoch, score=cur_model_score)

            if self.is_stop_early:
                logging.info("提前停止模型训练，长时间的模型效果没有得到提升!")
                break

    def train_epoch(self, epoch):
        self.net.train()

        for group_idx, param_group in enumerate(self.train_opt.param_groups):
            self.writer.add_scalar(f'lr_{group_idx}', param_group['lr'], global_step=epoch)

        for batch_x, batch_y in self.train_dataloader:
            self.train_step += 1
            scores = self.net(batch_x)  # 获取模型前向预测结果 [N,12]
            loss = self.loss_fn(scores, batch_y)  # 求解损失
            self.train_opt.zero_grad()  # 将每个参数的梯度重置为0
            loss.backward()  # 求解每个参数的梯度值
            self.train_opt.step()  # 参数更新
            self.writer.add_scalar('train_loss', loss.item(), global_step=self.train_step)

            if self.train_step % 100 == 0:
                logging.info(f"epoch {epoch}/{self.total_epoch} batch {self.train_step} loss:{loss.item():.3f}")

    def eval_epoch(self, epoch):
        self.net.eval()
        with torch.no_grad():
            test_y_true, test_y_pred = [], []
            for batch_x, batch_y in self.test_dataloader:
                scores = self.net(batch_x)  # 获取模型前向预测结果 [N,12]
                y_pred = torch.argmax(scores, dim=1)
                test_y_true.append(batch_y)
                test_y_pred.append(y_pred)

            test_y_true = torch.concatenate(test_y_true, dim=0).numpy()
            test_y_pred = torch.concatenate(test_y_pred, dim=0).numpy()

            confusion_matrix = metrics.confusion_matrix(test_y_true, test_y_pred)
            logging.info(f"eval epoch {epoch} confusion_matrix:\n{confusion_matrix}\n")
            report = metrics.classification_report(test_y_true, test_y_pred, zero_division=0.0, output_dict=True)
            logging.info(f"eval epoch {epoch} report:\n{report}\n")
            for label in report.keys():
                values = report[label]
                if isinstance(values, dict):
                    for metric_name in values.keys():
                        self.writer.add_scalar(
                            f'eval_{label}_{metric_name}', values[metric_name], global_step=epoch
                        )
                else:
                    self.writer.add_scalar(
                        f'eval_{label}', values, global_step=epoch
                    )

            if self.best_model_metric_func is None:
                return None
            else:
                return self.best_model_metric_func(test_y_true, test_y_pred)

    def save_epoch(self, epoch, score=None):
        """
        持久化模型保存
        保存的策略一：每调用一次，就保存一次
        保存的策略二：仅保存两个模型: best.pkl和last.pkl
        :param epoch:
        :param score: 效果评分
        :return:
        """
        if score is None:
            torch.save(
                {
                    'param': self.net.state_dict(),
                    'epoch': epoch
                },
                os.path.join(self.save_dir, f"model_{epoch:04d}.pkl")
            )
        else:
            save_model = {
                'param': self.net.state_dict(),
                'epoch': epoch,
                'score': score
            }
            torch.save(
                save_model,
                os.path.join(self.save_dir, f"last.pkl")
            )
            if score > self.best_score:
                logging.info(f"best model: {epoch} - {score} - {self.best_score}")
                torch.save(
                    save_model,
                    os.path.join(self.save_dir, "best.pkl")
                )
                self.best_score = score
                self.stop_early_epoch = 0
            elif self.stop_early:
                self.stop_early_epoch += 1  # 当前epoch的效果比上一个最优epoch差
                if self.stop_early_epoch > self.stop_early_epoch_interval:
                    self.is_stop_early = True
