# -*- coding: utf-8 -*-
"""
evaluate.py - 修复保存问题的评估版本
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
    """与推理脚本保持一致的文本处理"""
    if s is None:
        return ""
    s = str(s).strip()
    # 移除过多的空白字符，但不过度清洗
    s = re.sub(r'\s+', ' ', s)
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
            base_model: str = "chinese-llama-2-7b",
            lora_weights: str = "./lora-chinese-llama2-med/checkpoint-608",
            load_8bit: bool = False,
            device: str = "cuda",
            prompt_template: str = "med_template",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info("Loading tokenizer ...")

        # 与推理脚本完全一致的加载方式
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        logger.info("Loading base model (%s) ...", base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # 与推理脚本完全一致的tokenizer设置
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if lora_weights and os.path.exists(lora_weights):
            logger.info("Loading LoRA weights from %s ...", lora_weights)
            model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
        else:
            logger.warning("LoRA not found at %s – evaluating base model only.", lora_weights)

        if not load_8bit:
            model.half()
        model.eval()

        # 与推理脚本一致的编译设置
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.model = model

        # 使用相同的prompter
        from utils.prompter import Prompter
        self.prompter = Prompter(prompt_template)

    @torch.no_grad()
    def generate_one(self, instruction: str, input: Optional[str] = None) -> str:
        """与推理脚本完全一致的生成逻辑"""

        # 注意：这里不添加额外的instruction前缀，保持与推理脚本一致
        prompt = self.prompter.generate_prompt(instruction, input)

        # 1. 编码并去掉 token_type_ids - 与推理脚本一致
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        # 2. 使用与推理脚本相同的生成参数
        gen_config = GenerationConfig(
            temperature=0.1,  # 与推理脚本相同
            top_p=0.75,  # 与推理脚本相同
            top_k=40,  # 与推理脚本相同
            num_beams=1,  # 与推理脚本相同
            do_sample=True,  # 与推理脚本相同
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 3. 生成 - 与推理脚本相同的方式
        output_ids = self.model.generate(
            **inputs,
            generation_config=gen_config,
            max_new_tokens=512,  # 与推理脚本相同
        )

        # 4. 解码 - 与推理脚本相同的方式
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 5. 使用相同的response提取方法
        final_response = self.prompter.get_response(response)

        return final_response

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
                    logger.warning(f"Bad JSONL line {line_no}: {e}, skipping...")
                    continue
        return samples

    def eval_samples(
            self, samples: List[Dict[str, Any]], batch_size: int = 4, bleu_by_word: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        results, latencies = [], []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()

        success_count = 0
        for i, sample in enumerate(samples):
            try:
                t0 = time.perf_counter()

                pred = self.generate_one(sample["instruction"], sample.get("input"))
                golden = sample.get("output", "")

                # 标准化文本进行比较
                pred_norm = normalize_text(pred)
                gold_norm = normalize_text(golden)

                # 计算指标
                bleu = safe_bleu(gold_norm, pred_norm, use_word=bleu_by_word)
                rougeL = scorer.score(gold_norm, pred_norm)["rougeL"].fmeasure
                em = float(gold_norm == pred_norm)
                f1 = char_f1(gold_norm, pred_norm)

                # 记录结果 - 确保包含所有必要字段
                result_item = {
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "golden": golden,
                    "pred": pred,
                    "golden_normalized": gold_norm,  # 添加标准化后的文本用于调试
                    "pred_normalized": pred_norm,  # 添加标准化后的文本用于调试
                    "EM": em,
                    "F1": f1,
                    "BLEU": bleu,
                    "ROUGE-L": rougeL,
                }
                results.append(result_item)

                latency = time.perf_counter() - t0
                latencies.append(latency)
                success_count += 1

                # 打印每个样本的信息
                logger.info(f"样本 {i + 1}/{len(samples)}:")
                logger.info(f"  问题: {sample.get('instruction')}")
                logger.info(f"  参考答案: {golden}")
                logger.info(f"  模型回答: {pred}")
                logger.info(f"  指标 - EM: {em}, F1: {f1:.4f}, BLEU: {bleu:.4f}, ROUGE-L: {rougeL:.4f}")
                logger.info("-" * 80)

            except Exception as e:
                logger.error(f"处理样本 {i} 时出错: {e}")
                # 记录错误样本
                results.append({
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "golden": sample.get("output", ""),
                    "pred": f"ERROR: {str(e)}",
                    "golden_normalized": "",
                    "pred_normalized": "",
                    "EM": 0.0,
                    "F1": 0.0,
                    "BLEU": 0.0,
                    "ROUGE-L": 0.0,
                })

        total_time = time.perf_counter() - start

        if success_count > 0:
            # 只计算成功样本的指标
            successful_results = [r for r in results if not r["pred"].startswith("ERROR")]

            metrics = {
                "total_samples": len(samples),
                "successful_samples": success_count,
                "EM": sum(r["EM"] for r in successful_results) / len(successful_results),
                "F1": sum(r["F1"] for r in successful_results) / len(successful_results),
                "BLEU": sum(r["BLEU"] for r in successful_results) / len(successful_results),
                "ROUGE-L": sum(r["ROUGE-L"] for r in successful_results) / len(successful_results),
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "total_time": total_time,
                "samples/sec": success_count / total_time if total_time > 0 else 0,
            }
        else:
            metrics = {"error": "没有成功处理的样本"}

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
        """保存完整的结果到CSV和JSON文件"""
        try:
            # 创建DataFrame并保存为CSV
            df = pd.DataFrame(results)

            # 确保目录存在
            os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

            # 保存CSV文件
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info("成功保存CSV文件: %s", csv_path)
            logger.info("CSV文件包含 %d 行数据", len(df))

            # 保存JSON报告
            report_data = {
                "metrics": metrics,
                "samples": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_samples": len(results)
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            logger.info("成功保存JSON报告: %s", json_path)

            # 打印CSV文件的前几行作为验证
            logger.info("CSV文件前3行预览:")
            print(df.head(3).to_string())

        except Exception as e:
            logger.error("保存结果时出错: %s", e)
            raise


# ---------- CLI ----------
def main(
        base_model: str = "chinese-llama-2-7b",
        lora_weights: str = "./lora-chinese-llama2-med/checkpoint-608",
        test_path: str = "data/infer.json",
        output_csv: str = "eval_results.csv",
        output_json: str = "report.json",
        load_8bit: bool = False,
        batch_size: int = 1,  # 串行处理确保稳定性
        bleu_by_word: bool = True,
        use_lora: bool = True,
):
    logger.info("Starting evaluation with inference-compatible settings...")

    evaluator = EvaluateLlamaLoRA(
        base_model=base_model,
        lora_weights=lora_weights if use_lora else None,
        load_8bit=load_8bit,
    )

    if not os.path.exists(test_path):
        logger.error("测试文件不存在: %s", test_path)
        return

    samples = EvaluateLlamaLoRA.load_data(test_path)
    logger.info("Loaded %d samples from %s", len(samples), test_path)

    results, metrics = evaluator.eval_samples(
        samples,
        batch_size=batch_size,
        bleu_by_word=bleu_by_word
    )

    # 保存结果到文件
    evaluator.save_report(results, metrics, csv_path=output_csv, json_path=output_json)

    logger.info("评估完成!")
    logger.info("最终指标结果:")
    logger.info("  总样本数: %d", metrics["total_samples"])
    logger.info("  成功样本: %d", metrics["successful_samples"])
    logger.info("  EM: %.4f", metrics["EM"])
    logger.info("  F1: %.4f", metrics["F1"])
    logger.info("  BLEU: %.4f", metrics["BLEU"])
    logger.info("  ROUGE-L: %.4f", metrics["ROUGE-L"])
    logger.info("  平均延迟: %.2f秒", metrics["avg_latency"])
    logger.info("  生成速度: %.2f 样本/秒", metrics["samples/sec"])


if __name__ == "__main__":
    import fire

    fire.Fire(main)