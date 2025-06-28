# -*- coding = utf-8 -*-
# @File : main.py
# @Software : PyCharm
import functools
import logging

import torch
from torch.utils.data import DataLoader
from transformers import logging as tflogging

from METS.METS import METS
from train import ssl_train
from utils.dataset import SSLECGTextDataset, ZeroShotTestECGTextDataset
from utils.utils import get_smallest_loss_model_path, init_log
from zero_shot_classification import zero_shot_classification

# close BERT pretrain file loading warnings
tflogging.set_verbosity_error()

csv_path = "ptbxl_database.csv"
data_dir = "."

categories = ["NORM", "MI", "STTC", "CD", "HYP"]

if __name__ == "__main__":
    init_log()

    # 训练集
    train_dataset = SSLECGTextDataset(csv_path=csv_path, data_dir=data_dir, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 验证集
    val_dataset = SSLECGTextDataset(csv_path=csv_path, data_dir=data_dir, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    # 测试集（zero-shot 评估）
    test_dataset = ZeroShotTestECGTextDataset(csv_path=csv_path, data_dir=data_dir, categories=categories)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = METS(stage="train")
    loss_fn = functools.partial(model.contrastive_loss, tau=0.07)
    optimizer = torch.optim.Adam(model.ecg_encoder.parameters(), lr=0.001)
    # metrics_dict = {"acc": Accuracy(task="multiclass")}

    train_dfhistory = ssl_train(model,
                                optimizer,
                                loss_fn,
                                metrics_dict=None,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                epochs=50,
                                patience=5,
                                monitor="val_loss",
                                mode="min")
    logging.info("\n" + train_dfhistory.to_string())

    ckpt_path = get_smallest_loss_model_path("./checkpoint")
    test_dfhistory = zero_shot_classification(model, ckpt_path, test_dataset, test_dataloader)
    logging.info("\n" + test_dfhistory.to_string())
