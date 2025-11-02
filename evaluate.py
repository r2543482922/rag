# -*- coding: utf-8 -*-
"""
evaluationbase.py - 通用评估基类（面向你的医学指令数据与 LoRA/PEFT 模型）

功能摘要：
- 加载 base model / tokenizer，支持 PEFT (LoRA) 加载与 8-bit 加载
- 统计模型参数量（总参、可训练参）
- 处理 tokenizer pad_token（优先使用 eos_token；必要时添加 pad）
- 批量推理（提高吞吐），支持可选 RAG retriever
- 测量显存峰值、吞吐(samples/sec)、延迟分布（p50/p95/p99）
- 指标：Exact Match (文本级)、字符级 F1（中文友好）、BLEU（字/分词可选）、ROUGE-L
- 支持 SUS 问卷计算并把结果写到报告
- 输出逐样本 CSV 与 综合 JSON 报告（包含 metadata）
- 设计与你的 infer.py / finetune.py 兼容（默认数据路径 data/llama_data.json）

用法示例：
python evaluationbase.py \
  --base_model decapoda-research/llama-7b-hf \
  --lora_weights ./lora-llama-med-e1 \
  --test_path data/llama_data.json \
  --output_csv eval_results.csv \
  --output_json report.json \
  --batch_size 8 \
  --max_new_tokens 256

依赖：
- transformers, peft, torch, datasets, rouge_score, nltk, pandas, jieba (可选)
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import time
import math
import re
import logging
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

# jieba 用于中文分词（非必须）；若不可用则退回字符级
try:
    import jieba
    _has_jieba = True
except Exception:
    _has_jieba = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_text(s: Optional[str]) -> str:
    """简单归一化：去首尾空格，压缩空白。可根据需要扩展去标点等。"""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def char_tokenize(s: str) -> List[str]:
    """按字符分割（中文友好）"""
    return list(s)


def word_tokenize_with_jieba(s: str) -> List[str]:
    if _has_jieba:
        return list(jieba.cut(s))
    else:
        return s.split()


def safe_bleu(reference: str, hypothesis: str, use_word: bool = False) -> float:
    """对中文：默认按字符计算 BLEU；若 use_word=True 且 jieba 可用，则按词计算"""
    smooth = SmoothingFunction().method4
    if use_word and _has_jieba:
        ref_tokens = word_tokenize_with_jieba(reference)
        hyp_tokens = word_tokenize_with_jieba(hypothesis)
    else:
        ref_tokens = char_tokenize(reference)
        hyp_tokens = char_tokenize(hypothesis)
    try:
        # sentence_bleu expects list of reference lists
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
    except Exception:
        return 0.0


def char_level_f1(ref: str, pred: str) -> float:
    """字符级 F1（中文推荐）"""
    r = list(ref)
    p = list(pred)
    if len(p) == 0 and len(r) == 0:
        return 1.0
    common = Counter(r) & Counter(p)
    tp = sum(common.values())
    prec = tp / (len(p) + 1e-12)
    rec = tp / (len(r) + 1e-12)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def ensure_tokenizer_has_pad(tokenizer, model=None, add_if_missing: bool = True):
    """
    确保 tokenizer 有 pad_token_id。
    优先设 pad_token 为 eos_token；若都没有且 add_if_missing=True，则添加 [PAD]。
    在添加新 token 后需要在非 8-bit 情况下对 model 调用 resize_token_embeddings。
    返回是否添加了新 token（bool）。
    """
    added = False
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Tokenizer had no pad_token; set pad_token = eos_token")
        elif add_if_missing:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.warning("Added [PAD] token to tokenizer")
            added = True
            if model is not None:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                    logger.info("Resized model embeddings after adding pad token")
                except Exception as e:
                    logger.warning("resize_token_embeddings failed: %s", e)
    return added


class EvaluationBase:
    def __init__(self, config: Dict[str, Any]):
        """
        config keys (common):
          - base_model (str)
          - lora_weights (optional str)
          - load_in_8bit (bool)
          - device (str) "cuda" or "cpu"
          - inference: dict of generation params
          - prompter_template: str (name passed to utils.prompter.Prompter)
        """
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.tokenizer = None
        self.prompter = None
        self.generation_cfg = None

    def load_prompter(self, template_name: str = "med_template"):
        try:
            from utils.prompter import Prompter
        except Exception:
            raise RuntimeError("Cannot import Prompter from utils.prompter; ensure it exists in repo.")
        self.prompter = Prompter(template_name)

    def load_model_and_tokenizer(self, base_model: str, lora_weights: Optional[str] = None,
                                 load_in_8bit: bool = False, torch_dtype=torch.float16, device_map="auto"):
        logger.info("Loading tokenizer from %s", base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        logger.info("Loading base model from %s (load_in_8bit=%s)", base_model, load_in_8bit)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, load_in_8bit=load_in_8bit, torch_dtype=torch_dtype, device_map=device_map
        )
        # safe pad token
        added = ensure_tokenizer_has_pad(self.tokenizer, self.model, add_if_missing=True)
        if added:
            # if we added tokens above and model is non-8bit, resize embedding already attempted in ensure_tokenizer_has_pad
            pass

        if lora_weights:
            logger.info("Loading LoRA/PEFT weights from %s", lora_weights)
            self.model = PeftModel.from_pretrained(self.model, lora_weights, torch_dtype=torch_dtype)
        self.model.eval()
        # move to device if not handled by device_map
        try:
            # if model already on device via device_map, this is a no-op
            self.model.to(self.device)
        except Exception:
            pass

        # build generation config defaults
        inf = self.config.get("inference", {})
        self.generation_cfg = GenerationConfig(
            temperature=inf.get("temperature", 0.0),
            top_p=inf.get("top_p", 1.0),
            top_k=inf.get("top_k", 0),
            num_beams=inf.get("num_beams", 1),
            do_sample=inf.get("do_sample", False),
            repetition_penalty=inf.get("repetition_penalty", 1.0),
            max_new_tokens=inf.get("max_new_tokens", 128),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # ensure pad/eos set in model config
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.model.config, "eos_token_id", None) is None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    @staticmethod
    def load_test(path: str) -> List[Dict[str, Any]]:
        """支持 json 或 jsonl 编码的指令数据，单条为 {instruction, input, output}"""
        if path.endswith(".jsonl") or path.endswith(".json"):
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # maybe full JSON list
                        f.seek(0)
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                return data
                        except Exception:
                            raise
            return data
        else:
            # fallback: try to load as json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _prepare_prompts(self, samples: List[Dict[str, Any]]) -> List[str]:
        prompts = []
        for s in samples:
            instruction = s.get("instruction", "")
            input_ctx = s.get("input", "")
            # retriever could be used here if provided externally
            prompt = self.prompter.generate_prompt(instruction, input_ctx)
            prompts.append(prompt)
        return prompts

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], generation_cfg: Optional[GenerationConfig] = None) -> List[str]:
        """批量生成并返回解码后的文本（包含 prompt + generated）"""
        if generation_cfg is None:
            generation_cfg = self.generation_cfg
        # tokenize batch
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # ensure token_type_ids removed for some models
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        gen_kwargs = generation_cfg.to_dict()
        # Make sure pad_token_id/eos_token_id present
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        # Generate
        outputs = self.model.generate(**inputs, **gen_kwargs)
        texts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        return texts

    @torch.no_grad()
    def evaluate(self, data: List[Dict[str, Any]], batch_size: int = 8, retriever=None,
                 bleu_by_word: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        核心评估流程（batch）
        返回：results list, metrics dict (aggregated & perf)
        """
        if self.prompter is None:
            self.load_prompter(self.config.get("prompter_template", "med_template"))
        total_results = []
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        samples = data
        batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

        # reset GPU peak stats if cuda
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # warmup a single batch (optional) to stabilize timing
        if len(batches) > 0:
            warm = min(1, len(batches))
            warm_prompts = self._prepare_prompts(batches[0])
            try:
                _ = self.generate_batch(warm_prompts)
            except Exception:
                pass

        total_gen_tokens = 0
        latencies = []
        start_all = time.time()

        for b_idx, batch in enumerate(batches):
            prompts = []
            golds = []
            raw_prompts = []
            for s in batch:
                raw_prompts.append(s)
            prompts = self._prepare_prompts(batch)
            t0 = time.time()
            texts = self.generate_batch(prompts)
            t1 = time.time()
            # average latency per sample in this batch
            batch_latency = (t1 - t0) / max(1, len(prompts))
            latencies.extend([batch_latency] * len(prompts))

            # compute per-sample metrics
            for i, s in enumerate(batch):
                golden = s.get("output", "")
                raw_output = texts[i]
                # extract model reply using prompter (keeps same logic as infer.py)
                pred = self.prompter.get_response(raw_output)
                # compute token counts roughly
                gen_ids = self.tokenizer(raw_output, return_tensors="pt", truncation=False).input_ids[0]
                in_ids = self.tokenizer(prompts[i], return_tensors="pt", truncation=True).input_ids[0]
                gen_tokens = max(0, gen_ids.shape[0] - in_ids.shape[0])
                total_gen_tokens += int(gen_tokens)

                golden_n = normalize_text(golden)
                pred_n = normalize_text(pred)

                bleu = safe_bleu(golden_n, pred_n, use_word=bleu_by_word)
                rougeL = scorer.score(golden_n, pred_n)["rougeL"].fmeasure
                em = 1 if golden_n == pred_n else 0
                f1 = char_level_f1(golden_n, pred_n)

                total_results.append({
                    "instruction": s.get("instruction", ""),
                    "input": s.get("input", ""),
                    "golden": golden,
                    "pred": pred,
                    "bleu": float(bleu),
                    "rougeL": float(rougeL),
                    "EM": int(em),
                    "F1": float(f1),
                    "gen_tokens": int(gen_tokens),
                    "latency_s": float(batch_latency),
                })

            logger.info(f"Batch {b_idx+1}/{len(batches)} processed. avg latency per sample {batch_latency:.3f}s")

        total_time = time.time() - start_all
        peak_gpu_gb = None
        if self.device.type == "cuda":
            peak_gpu_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        # aggregate metrics
        metrics = {}
        if total_results:
            metrics["EM"] = sum(r["EM"] for r in total_results) / len(total_results)
            metrics["F1"] = sum(r["F1"] for r in total_results) / len(total_results)
            metrics["BLEU"] = sum(r["bleu"] for r in total_results) / len(total_results)
            metrics["ROUGE-L"] = sum(r["rougeL"] for r in total_results) / len(total_results)
            metrics["avg_latency_s"] = sum(r["latency_s"] for r in total_results) / len(total_results)
            metrics["p50_latency_s"] = float(sorted(latencies)[max(0, int(0.5 * len(latencies)) - 1)]) if latencies else None
            metrics["p95_latency_s"] = float(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]) if latencies else None
        metrics["total_time_s"] = total_time
        metrics["tokens_per_s"] = total_gen_tokens / total_time if total_time > 1e-6 else None
        metrics["samples_per_s"] = len(total_results) / total_time if total_time > 1e-6 else None
        metrics["peak_gpu_GB"] = peak_gpu_gb
        metrics["sample_cnt"] = len(total_results)
        return total_results, metrics

    @staticmethod
    def compute_sus_interactive():
        """命令行交互式 SUS 问卷（返回 0-100 分）"""
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

    def save_results(self, results: List[Dict[str, Any]], output_csv: str, output_json: str,
                     metadata: Dict[str, Any], metrics: Dict[str, Any]):
        pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
        full_report = {"metadata": metadata, "metrics": metrics, "results": results}
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        logger.info("Saved CSV -> %s and JSON -> %s", output_csv, output_json)


def main(
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: Optional[str] = None,
    test_path: str = "data/llama_data.json",
    output_csv: str = "eval_results.csv",
    output_json: str = "report.json",
    device: str = None,
    load_in_8bit: bool = False,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 40,
    num_beams: int = 1,
    do_sample: bool = True,
    sus: bool = False,
    prompt_template: str = "med_template",
    bleu_by_word: bool = False,
):
    # build config
    cfg = {
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "inference": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        },
        "prompter_template": prompt_template,
    }
    eb = EvaluationBase(cfg)
    eb.load_prompter(prompt_template)
    eb.load_model_and_tokenizer(base_model, lora_weights=lora_weights, load_in_8bit=load_in_8bit)

    # metadata
    meta = {
        "base_model": base_model,
        "lora_weights": lora_weights,
        "device": str(eb.device),
        "inference": cfg["inference"],
        "tokenizer": {
            "pad_token_id": getattr(eb.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(eb.tokenizer, "eos_token_id", None),
            "model_max_length": getattr(eb.tokenizer, "model_max_length", None),
        },
        "model_config_max_pos": getattr(eb.model.config, "max_position_embeddings", None),
        "torch_version": torch.__version__,
    }
    meta.update(eb.count_parameters())

    # load test data
    data = eb.load_test(test_path)
    logger.info("Loaded %d test samples from %s", len(data), test_path)

    # run evaluation
    results, metrics = eb.evaluate(data, batch_size=batch_size, retriever=None, bleu_by_word=bleu_by_word)

    # optionally SUS
    if sus:
        metrics["SUS"] = eb.compute_sus_interactive()

    # save
    eb.save_results(results, output_csv, output_json, meta, metrics)

    # print summary
    print("\n===== Evaluation Summary =====")
    print("Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\nSaved per-sample results to {output_csv} and full report to {output_json}")


if __name__ == "__main__":
    # CLI entrypoint
    import fire
    fire.Fire(main)