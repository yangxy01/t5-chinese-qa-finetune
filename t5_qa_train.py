import json
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
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

MAX_INPUT_LENGTH = 256   # 降低输入长度，减少内存占用（原512）
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 2           # 降低批大小，避免OOM（原8）
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    plt.show()
    print(f"收敛曲线已保存至：{save_path}")


# ===================== 训练一个 epoch =====================
def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    return total_loss / len(data_loader)


# ===================== 主训练流程 =====================
def main():
    print(f"使用设备：{DEVICE}")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    # 加载 tokenizer 和模型
    print(f"加载预训练模型：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # 构建数据集和 DataLoader
    print("加载训练数据集...")
    train_dataset = QADataset(TRAIN_DATA_PATH, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    dev_dataset = QADataset(DEV_DATA_PATH, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"训练集样本数：{len(train_dataset)}，验证集样本数：{len(dev_dataset)}")

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 开始训练
    train_losses = []
    dev_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        dev_loss = evaluate_one_epoch(model, dev_loader, DEVICE)

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")

    # 保存模型和 tokenizer
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"\n模型已保存至：{OUTPUT_MODEL_DIR}")

    # 绘制收敛曲线
    plot_loss_curve(train_losses, dev_losses, LOSS_CURVE_PATH)


if __name__ == "__main__":
    main()
