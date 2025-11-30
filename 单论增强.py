# -*- coding: utf-8 -*-

# python evaluate.py main --use_rag True --output_csv rag_results.csv --output_json rag_report.json
# python evaluate.py main --use_rag False --output_csv non_rag_results.csv --output_json non_rag_report.json
"""
evaluate.py - 支持 RAG/Non-RAG 切换的评估版本
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
from openai import OpenAI
from neo4j import GraphDatabase

try:
    import jieba

    _has_jieba = True
except Exception:
    _has_jieba = False

# 假设 Prompter 模块在 utils 文件夹中
try:
    from utils.prompter import Prompter
except ImportError:
    # 临时定义一个简化的 Prompter 以确保代码运行
    class Prompter:
        def __init__(self, template_name):
            pass

        def generate_prompt(self, instruction, input_context):
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n"

        def get_response(self, text):
            # 模仿原始代码的响应提取逻辑
            res = text.split("### Response:")[1].strip() if "### Response:" in text else text
            return res.split("</s>")[0].strip()


    print("Warning: Failed to import utils.prompter. Using simplified temporary Prompter.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Neo4j/NER 配置 (与推理代码保持一致) ----------
# 阿里云百炼 API Key
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "sk-00459b72ffb245e5958c40c595d8ff67")

# Neo4j 配置 (使用你提供的账号密码)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "lty20001114"  # 你的密码


# ---------- 1. 阿里云百炼 NER 模块 (从推理代码复制) ----------
class AliyunNERExtractor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.medical_dict = {'感冒', '高血压', '糖尿病', '冠心病', '胃炎', '头痛', '发烧', '咳嗽'}

    def extract_entities(self, text):
        """提取实体"""
        if not self.api_key: return self._local_extract(text)
        try:
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system",
                     "content": "你是一个医学实体识别专家。提取文本中的医学实体（疾病、症状、药品、检查）。只返回JSON列表，如[\"感冒\"]。"},
                    {"role": "user", "content": f"提取实体：{text}"}
                ],
                temperature=0.0
            )
            # ⚠️ 注意：这里根据推理代码中的异常内容进行了修正，以适应实际 API 响应
            txt = completion.choices[0].message.content.strip()
            clean_text = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"NER API Error: {e}. Falling back to local dictionary.")
            return self._local_extract(text)

    def _local_extract(self, text):
        return [w for w in self.medical_dict if w in text]


# ---------- 2. 自定义图谱检索器 (从推理代码复制) ----------
class MedicalGraphRetriever:
    def __init__(self, uri, user, pwd):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            logger.info("✅ Neo4j connected successfully for RAG evaluation.")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}. RAG retrieval will fail.")
            self.driver = None

    def query_entity_context(self, entity_name):
        if not self.driver: return ""

        context_parts = []
        with self.driver.session() as session:
            # 1. 查询是否是【疾病】节点 (Disease)
            q_disease_info = """
            MATCH (n:Disease {name: $name})
            RETURN n.desc AS desc, n.prevent AS prevent, n.cause AS cause, 
                   n.easy_get AS easy_get, n.cure_way AS cure_way
            """
            result = session.run(q_disease_info, name=entity_name).single()

            if result:
                info = result
                context_parts.append(f"【{entity_name}的基本信息】")
                if info['desc']: context_parts.append(f"简介：{info['desc']}")
                if info['cause']: context_parts.append(f"成因：{info['cause']}")
                # ... (省略其他属性，与原推理代码保持一致)

                # 2. 查询疾病的【关联关系】
                q_rels = """
                MATCH (n:Disease {name: $name})-[r]->(m)
                RETURN type(r) AS type, m.name AS target
                """
                rels = session.run(q_rels, name=entity_name)

                rel_dict = {}
                type_map = {
                    'recommand_eat': '推荐食谱', 'no_eat': '忌吃食物', 'do_eat': '宜吃食物',
                    'common_drug': '常用药品', 'recommand_drug': '推荐药品',
                    'need_check': '所需检查', 'has_symptom': '典型症状',
                    'acompany_with': '并发症', 'belongs_to': '所属科室'
                }

                for r in rels:
                    t = type_map.get(r['type'], r['type'])
                    if t not in rel_dict: rel_dict[t] = []
                    rel_dict[t].append(r['target'])

                for k, v in rel_dict.items():
                    context_parts.append(f"{k}：{'、'.join(v[:10])}")

            # 3. 查询【症状】节点
            q_symptom = """
            MATCH (n:Disease)-[:has_symptom]->(s:Symptom {name: $name})
            RETURN n.name AS disease LIMIT 10
            """
            res_sym = session.run(q_symptom, name=entity_name)
            diseases = [r['disease'] for r in res_sym]
            if diseases:
                context_parts.append(f"【{entity_name}】可能是以下疾病的症状：{'、'.join(diseases)}")

            # 4. 查询【药品】节点
            q_drug = """
            MATCH (d:Disease)-[:common_drug|recommand_drug]-(dr:Drug {name: $name})
            RETURN d.name AS disease LIMIT 10
            """
            res_drug = session.run(q_drug, name=entity_name)
            treated = [r['disease'] for r in res_drug]
            if treated:
                context_parts.append(f"【{entity_name}】常用于治疗：{'、'.join(treated)}")

        return "\n".join(context_parts)


# ---------- 工具 (保持不变) ----------
def normalize_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def char_tokenize(s):
    return list(s)


def safe_bleu(ref: str, hyp: str, use_word: bool = False) -> float:
    # ... (与原评估代码保持一致)
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
    # ... (与原评估代码保持一致)
    if not pred and not ref:
        return 1.0
    common = Counter(ref) & Counter(pred)
    tp = sum(common.values())
    prec = tp / (len(pred) + 1e-12)
    rec = tp / (len(ref) + 1e-12)
    return 2 * prec * rec / (prec + rec + 1e-12)


# ---------- 评估器 (添加 RAG 开关) ----------
class EvaluateLlamaLoRA:
    def __init__(
            self,
            base_model: str = "chinese-llama-2-7b",
            lora_weights: str = "./lora-chinese-llama2-med/checkpoint-608",
            load_8bit: bool = False,
            device: str = "cuda",
            prompt_template: str = "med_template",
            # 新增 RAG 开关
            use_rag: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_rag = use_rag
        logger.info("Loading tokenizer and model...")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if lora_weights and os.path.exists(lora_weights):
            logger.info("Loading LoRA weights from %s.", lora_weights)
            model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
        else:
            logger.warning("LoRA not found or disabled. Evaluating base model only.")

        if not load_8bit:
            model.half()
        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.model = model

        self.prompter = Prompter(prompt_template)

        # 3. 初始化 RAG 依赖 (仅在开启 RAG 时)
        if self.use_rag:
            logger.info("RAG mode enabled. Initializing NER and Neo4j Retriever...")
            self.ner_extractor = AliyunNERExtractor()
            self.kg_retriever = MedicalGraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PWD)
            self.rag_enabled = True
        else:
            logger.info("Non-RAG mode enabled. Skipping NER and Neo4j initialization.")
            self.rag_enabled = False

    @torch.no_grad()
    def generate_one(self, instruction: str, input: Optional[str] = None) -> str:

        question = f"{instruction} {input or ''}".strip()
        kg_context = ""

        # RAG 逻辑分支
        if self.rag_enabled:
            # 1. 实体提取
            entities = self.ner_extractor.extract_entities(question)

            # 2. 知识图谱检索
            if entities:
                contexts = []
                for ent in entities:
                    info = self.kg_retriever.query_entity_context(ent)
                    if info: contexts.append(info)
                kg_context = "\n\n".join(contexts)

        # 3. 构造 Prompt (RAG/Non-RAG 统一)
        if kg_context:
            input_context = (
                f"以下是检索到的医学知识库信息，请优先参考这些信息回答用户问题：\n"
                f"---------------------\n"
                f"{kg_context}\n"
                f"---------------------\n"
                f"用户补充信息：{input or '无'}"
            )
        else:
            # 如果不使用 RAG 或检索失败，则 input_context 仅包含用户补充信息
            input_context = input or ""

        prompt = self.prompter.generate_prompt(instruction, input_context)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        # 4. 生成
        gen_config = GenerationConfig(
            temperature=0.1,  # 与推理脚本相同
            top_p=0.75,  # 与推理脚本相同
            top_k=40,  # 与推理脚本相同
            num_beams=1,  # 与推理脚本相同
            do_sample=True,  # 确保能生成不同内容
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_ids = self.model.generate(
            **inputs,
            generation_config=gen_config,
            max_new_tokens=512,
        )

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        final_response = self.prompter.get_response(response)

        # 附加 RAG 状态，方便在报告中记录
        if self.rag_enabled:
            final_response = f"[RAG] {final_response}" if kg_context else f"[LLM] {final_response}"

        return final_response

    # --- (load_data, eval_samples, save_report 方法与原代码保持一致，略) ---
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
                logger.info(f"样本 {i + 1}/{len(samples)} (RAG: {self.rag_enabled}):")
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


# ---------- CLI (添加 RAG 开关) ----------
def main(
        base_model: str = "chinese-llama-2-7b",
        lora_weights: str = "./lora-chinese-llama2-med/checkpoint-608",
        test_path: str = "data/infer.json",
        output_csv: str = "eval_results.csv",
        output_json: str = "report.json",
        load_8bit: bool = False,
        batch_size: int = 1,
        bleu_by_word: bool = True,
        use_lora: bool = True,
        # 新增 RAG 开关
        use_rag: bool = True,
):
    """
    评估 Med-GRAG 系统。

    Args:
        use_rag (bool): 是否启用 RAG (NER + Neo4j) 机制。
        ... (其他参数)
    """
    mode = "RAG" if use_rag else "Non-RAG"
    logger.info("Starting evaluation in %s mode...", mode)

    evaluator = EvaluateLlamaLoRA(
        base_model=base_model,
        lora_weights=lora_weights if use_lora else None,
        load_8bit=load_8bit,
        use_rag=use_rag,  # 传递 RAG 开关
    )

    if not os.path.exists(test_path):
        logger.error("测试文件不存在: %s", test_path)
        return

    samples = EvaluateLlamaLoRA.load_data(test_path)
    logger.info("Loaded %d samples from %s", len(samples), test_path)

    # 修改输出文件名以区分 RAG/Non-RAG 结果
    base_name = os.path.splitext(output_csv)[0]
    output_csv = f"{base_name}_{mode.lower()}.csv"
    output_json = f"{base_name}_{mode.lower()}.json"

    results, metrics = evaluator.eval_samples(
        samples,
        batch_size=batch_size,
        bleu_by_word=bleu_by_word
    )

    # 保存结果到文件
    evaluator.save_report(results, metrics, csv_path=output_csv, json_path=output_json)

    logger.info("评估完成! 模式: %s", mode)
    logger.info("最终指标结果 (保存在 %s):", output_json)
    logger.info("  EM: %.4f", metrics["EM"])
    logger.info("  F1: %.4f", metrics["F1"])
    logger.info("  BLEU: %.4f", metrics["BLEU"])
    logger.info("  ROUGE-L: %.4f", metrics["ROUGE-L"])


if __name__ == "__main__":
    import fire

    fire.Fire(main)