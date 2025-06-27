# -*- coding = utf-8 -*-
# @File : zero_shot_classification.py
# @Software : PyCharm
import logging
import pandas as pd
import torch
from torchmetrics import Precision, Accuracy, Recall, F1Score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def zero_shot_classification(model, ckpt_path, test_dataset, test_dataloader):
    # ===== 加载模型参数（容错处理） =====
    print(f"✅ 加载模型参数: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    # ===== 模型准备 =====
    model = model.to(device)
    model.zero_shot_precess_text(test_dataset.categories)
    model.stage = "test"
    model.eval()

    # ===== 指标 =====
    precision = Precision(task="multiclass", num_classes=5).to(device)
    accuracy = Accuracy(task="multiclass", num_classes=5).to(device)
    recall = Recall(task="multiclass", num_classes=5).to(device)
    f1 = F1Score(task="multiclass", num_classes=5).to(device)

    test_history = {}
    logging.warning("=" * 25 + "Start Zero-Shot Testing" + "=" * 25)

    # ===== 推理 =====
    with torch.no_grad():
        predictions = []
        targets = []
        for i, (input, target) in enumerate(test_dataloader):
            input = input.to(device)
            target = target.to(device)

            max_probability_class = model(input, None)
            predictions.append(max_probability_class.flatten())
            targets.append(target.flatten())

        predictions = torch.cat(predictions).long()
        targets = torch.cat(targets).long()

        test_history["acc"] = accuracy(predictions, targets)
        test_history["pre"] = precision(predictions, targets)
        test_history["recall"] = recall(predictions, targets)
        test_history["f1"] = f1(predictions, targets)

    for key in test_history:
        test_history[key] = [test_history[key].item()]

    return pd.DataFrame(test_history, index=[0])
