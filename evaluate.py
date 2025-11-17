# -*- coding: utf-8 -*-
"""
evaluate.py  一键评估 LoRA 医学模型（nan/inf 修复版）
python evaluate.py \
  --base_model chinese-llama-2-7b \
  --lora_weights ./lora-chinese-llama2-med/checkpoint-608 \
  --test_path data/infer.json \
  --output_csv eval_chllama2.csv \
  --output_json report_chllama2.json \
  --bleu_by_word
"""
import os
import json
import time
import logging
import re
import sys
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    import jieba
    _has_jieba = True
except Exception:
    _has_jieba = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------- 工具 ----------
def normalize_text(s):
    s = re.sub(r"\s+", "", str(s).strip())
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)
    return s


def char_tokenize(s):
    return list(s)


def safe_bleu(ref: str, hyp: str, use_word: bool = False) -> float:
    smooth = SmoothingFunction().method4
    if use_word and _has_jieba:
        ref_t, hyp_t = list(jieba.cut(ref)), list(jieba.cut(hyp))
    else:
        ref_t, hyp_t = char_tokenize(ref), char_tokenize(hyp)
    try:
        return sentence_bleu([ref_t], hyp_t, smoothing_function=smooth)
    except Exception:
        return 0.0


def char_f1(ref: str, pred: str) -> float:
    if not pred and not ref:
        return 1.0
    common = Counter(ref) & Counter(pred)
    tp = sum(common.values())
    prec = tp / (len(pred) + 1e-12)
    rec = tp / (len(ref) + 1e-12)
    return 2 * prec * rec / (prec + rec + 1e-12)


# ---------- 评估器 ----------
class EvaluateLlamaLoRA:
    def __init__(
        self,
        base_model: str = "llama-7b",
        lora_weights: str = "./lora-llama-med",
        load_8bit: bool = False,
        device: str = "cuda",
        prompt_template: str = "med_template",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info("Loading tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

        logger.info("Loading base model (%s) ...", base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if lora_weights and os.path.exists(os.path.join(lora_weights, "adapter_model.bin")):
            logger.info("Loading LoRA weights from %s ...", lora_weights)
            model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
        else:
            logger.warning("LoRA not found / skipped – evaluating base model only.")

        if not load_8bit:
            model.half()
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.model = model

        from utils.prompter import Prompter
        self.prompter = Prompter(prompt_template)

        # 生成参数：温度略高 + FP32 推理
        self.gen_config = GenerationConfig(
            temperature=0.35,
            top_p=0.85,
            top_k=40,
            num_beams=2,
            max_new_tokens=256,
            do_sample=True,
            repetition_penalty=1.03,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    @torch.no_grad()
    def generate_one(self, instruction: str, input: Optional[str] = None) -> str:
        instruction = "你是医学助手，给出准确、简洁、安全的回答。\n" + instruction
        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)

        # 🔧 关键：生成阶段用 FP32，避免 FP16 溢出
        with torch.autocast(device_type=self.device.type, enabled=False):
            gen_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=self.gen_config,
                return_dict_in_generate=False,
                output_scores=False,
            )
        gen_text = self.tokenizer.decode(
            gen_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        return gen_text

    @staticmethod
    def load_data(path: str) -> List[Dict[str, Any]]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Bad JSONL line {line_no}: {e}\n{line[:80]}")
        return samples

    def eval_samples(
        self, samples: List[Dict[str, Any]], batch_size: int = 8, bleu_by_word: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        results, latencies = [], []

        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            t0 = time.perf_counter()
            for s in batch:
                pred = self.generate_one(s["instruction"], s.get("input"))
                golden = s.get("output", "")
                pred_n, gold_n = normalize_text(pred), normalize_text(golden)

                bleu = safe_bleu(gold_n, pred_n, use_word=bleu_by_word)
                rougeL = scorer.score(gold_n, pred_n)["rougeL"].fmeasure
                em = float(gold_n == pred_n)
                f1 = char_f1(gold_n, pred_n)

                results.append(
                    {
                        "instruction": s.get("instruction"),
                        "input": s.get("input"),
                        "golden": golden,
                        "pred": pred,
                        "EM": em,
                        "F1": f1,
                        "BLEU": bleu,
                        "ROUGE-L": rougeL,
                    }
                )
            latencies.append((time.perf_counter() - t0) / len(batch))

        total_time = time.perf_counter() - start
        metrics = {
            "samples": len(results),
            "EM": sum(r["EM"] for r in results) / len(results),
            "F1": sum(r["F1"] for r in results) / len(results),
            "BLEU": sum(r["BLEU"] for r in results) / len(results),
            "ROUGE-L": sum(r["ROUGE-L"] for r in results) / len(results),
            "avg_latency": sum(latencies) / len(latencies),
            "total_time": total_time,
            "samples/sec": len(results) / total_time,
        }
        if self.device.type == "cuda":
            metrics["peak_gpu_GB"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
        return results, metrics

    def save_report(
        self,
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        csv_path: str = "eval_results.csv",
        json_path: str = "report.json",
    ):
        pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "samples": results}, f, ensure_ascii=False, indent=2)
        logger.info("Saved %s & %s", csv_path, json_path)


# ---------- CLI ----------
def main(
    base_model: str = "llama-7b",
    lora_weights: str = "./lora-llama-med",
    test_path: str = "data/infer.json",
    output_csv: str = "eval_results.csv",
    output_json: str = "report.json",
    load_8bit: bool = False,
    batch_size: int = 8,
    bleu_by_word: bool = True,
    use_lora: bool = True,
):
    logger.info("Starting evaluation …")
    evaluator = EvaluateLlamaLoRA(
        base_model=base_model,
        lora_weights=lora_weights if use_lora else None,
        load_8bit=load_8bit,
    )
    samples = EvaluateLlamaLoRA.load_data(test_path)
    logger.info("Loaded %d samples from %s", len(samples), test_path)
    results, metrics = evaluator.eval_samples(samples, batch_size=batch_size, bleu_by_word=bleu_by_word)
    evaluator.save_report(results, metrics, csv_path=output_csv, json_path=output_json)
    logger.info("metrics: %s", metrics)


if __name__ == "__main__":
    import fire
    fire.Fire(main)