#!/usr/bin/env python3
"""
医学大模型综合测评脚本
用法：
python eval_med.py \
  --base_model hfl/chinese-llama-2-7b \
  --lora_weights ./lora-chinese-llama2-med-e5 \
  --test_path data/med_test.jsonl \
  --output_csv report.csv \
  --sus  # 如需 SUS 问卷
"""
import fire
import json
import pandas as pd
import time
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftModel
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

from utils.prompter import Prompter  # 沿用你的模板

try:
    from retriever import MedRetriever
except:
    MedRetriever = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_test(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f]


def sus_survey():
    print("【系统可用性量表 SUS】请对 1-5 打分（1=非常不同意，5=非常同意）")
    questions = [
        "我愿意经常使用这个系统",
        "我觉得系统过于复杂",
        "我觉得系统易于使用",
        "我需要技术人员支持才能使用",
        "我发现各种功能整合得很好",
        "我觉得系统里有很多不一致",
        "我认为大多数人能快速上手",
        "我觉得系统非常繁琐",
        "我使用系统时很有信心",
        "我需要学习很多才能开始",
    ]
    score = 0
    for i, q in enumerate(questions, 1):
        while True:
            try:
                s = int(input(f"{i:02}. {q} ："))
                if 1 <= s <= 5:
                    score += (s - 1) if i % 2 == 0 else (5 - s)
                    break
                else:
                    print("请输入 1-5 之间的整数")
            except ValueError:
                print("输入无效，请重新输入")
    return score * 2.5  # 0-100


@torch.no_grad()
def evaluate(model, tokenizer, prompter, data, generation_config, rag_ret=None):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method4
    results = []
    total_tokens = 0
    start = time.time()
    max_gpu = 0

    for idx, sample in enumerate(data):
        instruction = sample["instruction"]
        golden = sample["output"]
        input_ctx = sample.get("input", "")

        # RAG
        knowledge = rag_ret.retrieve(instruction) if rag_ret else ""
        if knowledge:
            input_ctx = (knowledge + "\n\n" + input_ctx).strip()

        prompt = prompter.generate_prompt(instruction, input_ctx)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        # 推理
        tic = time.time()
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=generation_config.max_new_tokens,
        )
        latency = time.time() - tic
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred = prompter.get_response(pred)

        # 指标计算
        total_tokens += output_ids.shape[1] - inputs.input_ids.shape[1]
        max_gpu = max(max_gpu, torch.cuda.max_memory_allocated() / 1024 ** 3)

        bleu = sentence_bleu([golden.split()], pred.split(), smoothing_function=smooth)
        rougeL = scorer.score(golden, pred)["rougeL"].fmeasure

        results.append({
            "instruction": instruction,
            "golden": golden,
            "pred": pred,
            "bleu": bleu,
            "rougeL": rougeL,
            "latency": latency,
        })
        print(f"[{idx + 1}/{len(data)}]  BLEU={bleu:.3f}  ROUGE-L={rougeL:.3f}  latency={latency:.2f}s")

    total_time = time.time() - start
    return results, total_tokens, total_time, max_gpu


def compute_metrics(results):
    preds = [r["pred"] for r in results]
    golds = [r["golden"] for r in results]
    # 简单 Exact-Match 准确率
    em = accuracy_score(golds, preds)
    # F1 基于字符级
    f1 = f1_score(golds, preds, average="micro")
    bleu = sum(r["bleu"] for r in results) / len(results)
    rouge = sum(r["rougeL"] for r in results) / len(results)
    return {"EM": em, "F1": f1, "BLEU": bleu, "ROUGE-L": rouge}


def main(
        base_model: str = "hfl/chinese-llama-2-7b",
        lora_weights: str = "./lora-chinese-llama2-med-e5",
        test_path: str = "data/med_test.jsonl",
        output_csv: str = "report.csv",
        output_json: str = "report.json",
        rag: bool = False,
        sus: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.75,
        top_k: int = 40,
        num_beams: int = 4,
        max_new_tokens: int = 256,
):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, lora_weights)
    model.eval()
    prompter = Prompter("med_template")

    # ---------- RAG ----------
    retriever = MedRetriever(index_path="med.index") if rag and MedRetriever else None

    gen_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    data = load_test(test_path)
    results, tokens, total_time, max_gpu = evaluate(
        model, tokenizer, prompter, data, gen_config, retriever
    )

    metrics = compute_metrics(results)
    metrics["tokens/s"] = tokens / total_time
    metrics["max_gpu_GB"] = max_gpu
    metrics["total_time_s"] = total_time
    metrics["sample_cnt"] = len(data)

    # ---------- SUS ----------
    if sus:
        print("\n===== SUS 问卷 =====")
        metrics["SUS"] = sus_survey()

    # ---------- 输出 ----------
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "samples": results}, f, ensure_ascii=False, indent=2)

    print("\n===== 综合报告 =====")
    for k, v in metrics.items():
        print(f"{k:15s}: {v}")
    print(f"详细结果已保存至 {output_csv} 与 {output_json}")


if __name__ == "__main__":
    fire.Fire(main)
