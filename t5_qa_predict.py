import json
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ===================== 配置参数 =====================
MODEL_DIR = "output/t5_qa_model"
DEV_DATA_PATH = "data/dev.json"
MAX_INPUT_LENGTH = 512
MAX_GENERATE_LENGTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 加载模型 =====================
def load_model(model_dir: str):
    """从指定目录加载已训练好的 tokenizer 和模型"""
    print(f"加载模型：{model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# ===================== 单条预测 =====================
def predict_answer(context: str, question: str, tokenizer, model) -> str:
    """
    给定任意 context 和 question，生成对应答案。

    Args:
        context:  参考文章内容
        question: 问题文本
        tokenizer: 已加载的 tokenizer
        model:    已加载的 T5 模型

    Returns:
        生成的答案字符串
    """
    input_text = f"问题：{question} 文章：{context}"
    inputs = tokenizer(
        input_text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_GENERATE_LENGTH,
            num_beams=4,
            early_stopping=True,
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer


# ===================== BLEU 评估 =====================
def evaluate_bleu_on_dev(tokenizer, model, dev_data_path: str):
    """
    在验证集上计算 BLEU-1 ~ BLEU-4 指标。

    Args:
        tokenizer:     已加载的 tokenizer
        model:         已加载的 T5 模型
        dev_data_path: 验证集文件路径
    """
    references = []   # 参考答案列表，每个元素为 [list_of_tokens]
    hypotheses = []   # 预测答案列表，每个元素为 list_of_tokens

    with open(dev_data_path, "r", encoding="utf-8") as file:
        samples = [json.loads(line.strip()) for line in file if line.strip()]

    print(f"验证集共 {len(samples)} 条样本，开始预测...")

    for item in tqdm(samples, desc="BLEU Evaluation"):
        predicted_answer = predict_answer(item["context"], item["question"], tokenizer, model)
        # 按字符级别分词（适合中文）
        reference_tokens = list(item["answer"])
        hypothesis_tokens = list(predicted_answer)
        references.append([reference_tokens])
        hypotheses.append(hypothesis_tokens)

    smoothing = SmoothingFunction().method1

    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    print("\n========== BLEU 评估结果 ==========")
    print(f"BLEU-1: {bleu_1:.4f}")
    print(f"BLEU-2: {bleu_2:.4f}")
    print(f"BLEU-3: {bleu_3:.4f}")
    print(f"BLEU-4: {bleu_4:.4f}")

    return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}


# ===================== 交互式预测演示 =====================
def interactive_predict(tokenizer, model):
    """交互式命令行问答演示"""
    print("\n========== 交互式问答演示（输入 'exit' 退出）==========")
    while True:
        context = input("\n请输入参考文章（context）：").strip()
        if context.lower() == "exit":
            break
        question = input("请输入问题（question）：").strip()
        if question.lower() == "exit":
            break

        answer = predict_answer(context, question, tokenizer, model)
        print(f"模型生成答案：{answer}")


# ===================== 主流程 =====================
def main():
    print(f"使用设备：{DEVICE}")

    tokenizer, model = load_model(MODEL_DIR)

    # 在验证集上进行 BLEU 评估
    evaluate_bleu_on_dev(tokenizer, model, DEV_DATA_PATH)

    # 交互式预测演示
    interactive_predict(tokenizer, model)


if __name__ == "__main__":
    main()
