# -*- coding = utf-8 -*-
# @File : utils.py
# @Software : PyCharm

import datetime
import logging
import os
import random
import re

import numpy as np
import torch


def epoch_log(info):
    logging.info("Start A New Epoch" + "\n" + "==========" * 8)
    logging.info(str(info) + "\n")


def init_log():
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f'logs/{nowtime}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_loss_from_filename(filename):
    """ 从 ckpt 文件名中提取 val_loss 数值 """
    match = re.search(r'val_loss=([0-9.]+)\.pt', filename)
    return float(match.group(1)) if match else None


def keep_top_files(folder_path, top_x):
    """ 保留前 top_x 个最小 val_loss 的 ckpt 文件，删除其余 """
    files = os.listdir(folder_path)
    files_with_loss = []
    for file in files:
        loss = extract_loss_from_filename(file)
        if loss is not None:
            files_with_loss.append((file, loss))
        else:
            logging.warning(f"⚠️ 未识别损失值，跳过文件: {file}")

    if not files_with_loss:
        logging.error("❌ 未找到有效的 val_loss 文件，无法排序。")
        return

    files_sorted = sorted(files_with_loss, key=lambda x: x[1], reverse=True)
    for file, _ in files_sorted[top_x:]:
        os.remove(os.path.join(folder_path, file))
        logging.warning(f"🗑 删除文件: {file}")


def get_smallest_loss_model_path(folder_path):
    """ 获取 val_loss 最小模型的 ckpt 路径 """
    files = os.listdir(folder_path)
    files_with_loss = [(file, extract_loss_from_filename(file)) for file in files]
    files_with_loss = [file for file in files_with_loss if file[1] is not None]

    if not files_with_loss:
        logging.error("❌ 未找到任何带 val_loss 的模型文件！")
        return None

    files_sorted = sorted(files_with_loss, key=lambda x: x[1])
    smallest_loss_file = files_sorted[0][0]
    return os.path.join(folder_path, smallest_loss_file)
