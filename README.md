# T5 中文问答微调（t5-chinese-qa-finetune）

基于 [mengzi-t5-base](https://huggingface.co/langboat/mengzi-t5-base) 预训练模型，在中文阅读理解数据集上进行微调，实现给定文章和问题后自动生成答案的能力。

---

## 项目结构

```
t5-chinese-qa-finetune/
├── data/
│   ├── train.json          # 训练集（约 14520 条）
│   └── dev.json            # 验证集（约 984 条）
├── output/
│   └── t5_qa_model/        # 训练完成后保存的模型权重
├── t5_qa_train.py          # 模型训练脚本
├── t5_qa_predict.py        # 模型推理 & BLEU 评估脚本
└── README.md
```

---

## 数据格式

`train.json` 和 `dev.json` 均为 **每行一条 JSON** 的格式，每条样本包含以下字段：

```json
{
  "context":  "参考文章内容...",
  "question": "问题文本...",
  "answer":   "标准答案..."
}
```

---

## 环境依赖

```bash
pip install torch transformers nltk matplotlib tqdm
```

> 推荐使用 GPU 环境训练，CPU 训练速度较慢。

---

## 使用方式

### 第一步：训练模型

```bash
python t5_qa_train.py
```

训练过程说明：
- 自动从 HuggingFace 下载 `langboat/mengzi-t5-base` 预训练权重
- 训练 3 个 epoch，每个 epoch 结束后在验证集上计算 loss
- 训练完成后模型保存至 `output/t5_qa_model/`
- 同时生成 loss 收敛曲线图 `output/loss_curve.png`

主要训练参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_INPUT_LENGTH` | 256 | 输入序列最大长度 |
| `MAX_TARGET_LENGTH` | 64 | 目标序列最大长度 |
| `BATCH_SIZE` | 2 | 批大小（内存不足时可调小） |
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `LEARNING_RATE` | 3e-4 | 学习率 |

---

### 第二步：推理与评估

```bash
python t5_qa_predict.py
```

推理脚本会依次执行两件事：

1. **BLEU 评估**：在 `data/dev.json` 验证集上计算 BLEU-1 ~ BLEU-4 指标，评估模型生成质量
2. **交互式问答**：进入命令行交互模式，输入任意文章和问题，模型实时生成答案

交互示例：

```
请输入参考文章（context）：北京是中国的首都，位于华北平原北部...
请输入问题（question）：北京位于哪里？
模型生成答案：华北平原北部
```

> 输入 `exit` 可退出交互模式。

---

## 注意事项

- **内存不足（Killed）**：将 `t5_qa_train.py` 中的 `BATCH_SIZE` 调小（如改为 1），或增加系统 swap 空间
- **首次运行**：会自动从 HuggingFace 下载约 990MB 的模型权重，需要保证网络畅通
- **数据文件**：`data/` 目录已加入 `.gitignore`，不会上传到 GitHub，需自行准备数据
