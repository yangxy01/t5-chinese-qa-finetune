import json
import os
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，无需 GUI
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# ===================== 配置参数 =====================
MODEL_NAME = "langboat/mengzi-t5-base"
TRAIN_DATA_PATH = "data/train.json"
DEV_DATA_PATH = "data/dev.json"
OUTPUT_MODEL_DIR = "output/t5_qa_model"
LOSS_CURVE_PATH = "output/loss_curve.png"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 64          # H20 显存充足，大幅提升 batch size 加速训练
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.1
NUM_WORKERS = 4          # DataLoader 多进程加载数据
USE_AMP = True           # 启用混合精度训练 (FP16)，显著加速
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()


# ===================== 数据集定义 =====================
class QADataset(Dataset):
    """问答数据集，将 context + question 拼接作为输入，answer 作为目标输出"""

    def __init__(self, data_path: str, tokenizer, max_input_length: int, max_target_length: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        # 构造输入文本：问题 + 参考文章
        input_text = f"问题：{item['question']} 文章：{item['context']}"
        target_text = item["answer"]

        model_inputs = self.tokenizer(
            input_text,
            text_target=target_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        # 将 padding token 替换为 -100，使其在 loss 计算中被忽略
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


# ===================== 绘制收敛曲线 =====================
def plot_loss_curve(train_losses: list, dev_losses: list, save_path: str):
    """绘制训练集和验证集的 loss 收敛曲线并保存"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, dev_losses, marker="s", label="Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("T5 QA Model Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"收敛曲线已保存至：{save_path}")


# ===================== 训练一个 epoch =====================
def train_one_epoch(model, data_loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # 比 zero_grad() 更高效

        if scaler is not None:
            # 混合精度前向 + 反向
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                # DataParallel 多卡时 loss 会是多维的，取均值
                if loss.dim() > 0:
                    loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


# ===================== 验证一个 epoch =====================
def evaluate_one_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if USE_AMP:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    if loss.dim() > 0:
                        loss = loss.mean()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()

            total_loss += loss.item()

    return total_loss / len(data_loader)


# ===================== 主训练流程 =====================
def main():
    print(f"使用设备：{DEVICE}，可用 GPU 数量：{NUM_GPUS}")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    # 加载 tokenizer 和模型
    print(f"加载预训练模型：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # 多 GPU 并行
    if NUM_GPUS > 1:
        print(f"启用 DataParallel，使用 {NUM_GPUS} 张 GPU")
        model = nn.DataParallel(model)

    # 构建数据集和 DataLoader（pin_memory + 多 worker 加速数据加载）
    print("加载训练数据集...")
    train_dataset = QADataset(TRAIN_DATA_PATH, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    dev_dataset = QADataset(DEV_DATA_PATH, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"训练集样本数：{len(train_dataset)}，验证集样本数：{len(dev_dataset)}")
    print(f"Batch Size: {BATCH_SIZE}，混合精度(AMP): {USE_AMP}，DataLoader Workers: {NUM_WORKERS}")

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 混合精度 scaler
    scaler = GradScaler() if USE_AMP else None

    # 开始训练
    train_losses = []
    dev_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE, scaler)
        dev_loss = evaluate_one_epoch(model, dev_loader, DEVICE)

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")

    # 保存模型和 tokenizer（处理 DataParallel 包装）
    save_model = model.module if isinstance(model, nn.DataParallel) else model
    save_model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"\n模型已保存至：{OUTPUT_MODEL_DIR}")

    # 绘制收敛曲线
    plot_loss_curve(train_losses, dev_losses, LOSS_CURVE_PATH)


if __name__ == "__main__":
    main()
