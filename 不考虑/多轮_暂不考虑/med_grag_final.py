# med_grag_final.py

# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
Med-GRAG Final: Chinese-LLaMA-2 + LoRA + Qwen-NER + Custom Medical Graph
- 修复 Runtime Error: 彻底移除 bitsandbytes 量化，改用原生 FP16
- 修复 RAG逻辑: 实体优先级 + 强制回答当前问题 + 提高防重复惩罚
"""
import sys
import os
import json
import re
import fire
import gradio as gr
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from neo4j import GraphDatabase

# 导入 utils/prompter.py
try:
    from utils.prompter import Prompter
except ImportError:
    print("❌ 错误: 无法导入 utils.prompter。请确保 utils/prompter.py 文件存在。")
    sys.exit(1)

# ==========================================
# 🛑 环境配置
# ==========================================

# 1. 解决 OOM 的碎片化问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 2. 阿里云百炼 API Key
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "sk-00459b72ffb245e5958c40c595d8ff67")

# ==========================================
# 配置区域
# ==========================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "lty20001114"

BASE_MODEL_PATH = "chinese-llama-2-7b"
LORA_WEIGHTS_PATH = "lora-chinese-llama2-med/checkpoint-608"


# ---------- 1. 阿里云百炼 NER 模块 (保持不变) ----------
class AliyunNERExtractor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("⚠️ WARN: DASHSCOPE_API_KEY not found. Using local dictionary.")
        self.client = OpenAI(api_key=self.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.medical_dict = {'感冒', '高血压', '糖尿病', '冠心病', '胃炎', '头痛', '发烧', '咳嗽', '阿司匹林', '失眠'}

    def extract_entities(self, text):
        if not self.api_key: return self._local_extract(text)
        try:
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system",
                     "content": "你是一个医学实体识别专家。提取所有医学实体，只返回JSON格式的列表，如[\"感冒\"]。"},
                    {"role": "user", "content": f"提取实体：{text}"}
                ],
                temperature=0.0
            )
            txt = completion.choices[0].message.content.strip()
            clean_text = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"❌ NER API Error: {e}")
            return self._local_extract(text)

    def _local_extract(self, text):
        return [w for w in self.medical_dict if w in text]


# ---------- 2. 自定义图谱检索器 (优化版) ----------
class MedicalGraphRetriever:
    def __init__(self, uri, user, pwd):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self.driver.verify_connectivity()
            print("✅ Neo4j connected successfully.")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            self.driver = None

    RELATION_MAP = {
        'recommand_eat': '推荐食谱', 'no_eat': '忌吃食物', 'do_eat': '宜吃食物',
        'common_drug': '常用药品', 'recommand_drug': '推荐药品',
        'need_check': '所需检查', 'has_symptom': '典型症状',
        'acompany_with': '并发症', 'belongs_to': '所属科室'
    }

    def query_entity_context(self, entity_name):
        if not self.driver: return ""
        context_parts = []

        # 过滤掉一些无用的通用实体
        stop_words = {'内科', '外科', '医院', '医生', '建议', '检查'}
        if entity_name in stop_words:
            return ""

        with self.driver.session() as session:
            # 1. 优先查询【药品】(Drug)
            q_drug_node = "MATCH (n:Drug {name: $name}) RETURN n.desc AS desc, n.effect AS effect"
            result_drug = session.run(q_drug_node, name=entity_name).data()
            if result_drug:
                info = result_drug[0]
                context_parts.append(f"【药品：{entity_name}】")
                if info.get('desc'): context_parts.append(f"说明：{info['desc'][:100]}...")

                q_drug_cure = "MATCH (n:Drug {name: $name})-[:recommand_drug|common_drug]-(d:Disease) RETURN d.name as disease LIMIT 5"
                cures = [r['disease'] for r in session.run(q_drug_cure, name=entity_name)]
                if cures: context_parts.append(f"主治疾病：{'、'.join(cures)}")

            # 2. 查询【疾病】(Disease)
            q_disease_info = """
            MATCH (n:Disease {name: $name})
            RETURN n.desc AS desc, n.prevent AS prevent, n.cause AS cause, n.cure_way AS cure_way
            """
            result_disease = session.run(q_disease_info, name=entity_name).data()
            if result_disease:
                info = result_disease[0]
                context_parts.append(f"【疾病：{entity_name}】")
                if info.get('desc'): context_parts.append(f"简介：{info['desc'][:100]}...")

                # 重点：饮食和药物查询
                q_rels = "MATCH (n:Disease {name: $name})-[r]->(m) RETURN type(r) AS type, m.name AS target"
                rels = session.run(q_rels, name=entity_name)

                rel_dict = {}
                for r in rels:
                    t = self.RELATION_MAP.get(r['type'], r['type'])
                    if t not in rel_dict: rel_dict[t] = []
                    rel_dict[t].append(r['target'])

                # 优先展示饮食和药物，并限制数量
                priority_keys = ['忌吃食物', '宜吃食物', '推荐食谱', '常用药品', '推荐药品']
                for k in priority_keys:
                    if k in rel_dict:
                        context_parts.append(f"{k}：{'、'.join(rel_dict[k][:5])}")  # 限制只显示5个

        # 限制总 Context 长度
        return "\n".join(context_parts[:8])

    # 初始化组件


ner_extractor = AliyunNERExtractor()
kg_retriever = MedicalGraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PWD)

# ---------- 3. 模型推理主逻辑 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(
        base_model: str = BASE_MODEL_PATH,
        use_lora: bool = True,
        lora_weights: str = LORA_WEIGHTS_PATH,
        prompt_template: str = "med_template",
        gradio: bool = False,
):
    prompter = Prompter(prompt_template)

    # 尝试加载 LLM
    actual_evaluate = None
    try:
        print(f"⏳ 正在加载模型 (FP16模式)...")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        # ⚠️ 关键修复：使用 torch.float16 加载，解决 Quantization shape 错误
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if use_lora:
            print(f"⏳ 正在加载 LoRA: {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16
            )

        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.eval()
        print("✅ LLM 加载成功 (FP16 Native)!")

    except Exception as e:
        print(f"❌ Error loading LLM: {e}")

        # Mock 函数 (兜底)
        def mock_evaluate(instruction, history, **kwargs):
            formatted_history = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
            full_context = f"{formatted_history}\n用户：{instruction}"
            entities = ner_extractor.extract_entities(full_context)
            kg_context = ""
            if entities:
                kg_context = kg_retriever.query_entity_context(entities[0])

            if kg_context:
                mock_answer = f"【模拟RAG回答 (模型加载失败)】根据'{entities[0]}'，检索到信息。请问您还有其他症状吗？"
            else:
                mock_answer = f"【模拟回答 (模型加载失败)】您提到了'{instruction}'。请您详细描述一下您最近感觉如何？"
            return mock_answer, history + [(instruction, mock_answer)]

        actual_evaluate = mock_evaluate
        print("⚠️ 已切换到 Mock 模式。")

    # 历史记录格式化
    def format_history(history):
        formatted = []
        for user_q, bot_a in history:
            formatted.append(f"用户：{user_q}")
            formatted.append(f"助手：{bot_a}")
        return "\n".join(formatted)

    @torch.no_grad()
    def evaluate(instruction, history, **kwargs):
        # 如果模型加载失败，调用 mock
        if actual_evaluate:
            return actual_evaluate(instruction, history, **kwargs)

        if device == "cuda":
            torch.cuda.empty_cache()

        current_question = instruction.strip()

        # A. 提取实体：区分“当前问题实体”和“历史实体”
        current_entities = ner_extractor.extract_entities(current_question)
        formatted_history_text = "\n".join([f"{u} {b}" for u, b in history])
        history_entities = ner_extractor.extract_entities(formatted_history_text)

        # 合并实体：当前实体优先，然后是历史中未被当前问题提及的实体
        final_search_entities = current_entities + [e for e in history_entities if e not in current_entities]

        print(f"🔍 [Current Entities]: {current_entities}")
        print(f"🔍 [History Entities]: {history_entities}")

        # B. 知识图谱检索
        kg_context = ""
        if final_search_entities:
            contexts = []
            # 只查前 3 个实体
            for ent in final_search_entities[:3]:
                info = kg_retriever.query_entity_context(ent)
                if info: contexts.append(info)
            kg_context = "\n\n".join(contexts)

        print(f"📚 [Graph Context]:\n{kg_context[:100]}..." if kg_context else "📚 [Graph Context]: None")

        # C. 构造 Prompt (强化指令，防止复读)
        formatted_history = format_history(history)

        # ⚠️ 强化系统指令
        system_prompt = (
            "你是一名专业的医生。请基于【知识库信息】回答用户的【当前问题】。\n"
            "注意：\n"
            "1. 如果用户问饮食，必须回答忌口和宜吃食物。\n"
            "2. 如果用户问药物，必须基于知识库说明药物作用和适应症。\n"
            "3. **禁止重复之前的回答**，必须针对【当前问题】进行新一轮的解答。"
        )

        input_context = (
            f"【知识库信息】:\n{kg_context or '暂无具体数据，请依据常识回答'}\n\n"
            f"【历史对话】:\n{formatted_history}\n\n"
            f"【当前问题】:\n{current_question}"
        )

        prompt = prompter.generate_prompt(
            instruction=system_prompt,
            input=input_context
        )

        # D. 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)

        output_ids = model.generate(
            **inputs,
            generation_config=GenerationConfig(
                temperature=0.2,  # 稍微提高，避免死板
                top_p=0.8,
                top_k=40,
                num_beams=1,
                max_new_tokens=512,
                repetition_penalty=1.2  # 关键！提高防复读惩罚
            )
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        final_answer = prompter.get_response(response)

        # 5. 更新历史记录
        new_history = history + [(current_question, final_answer)]
        return final_answer, new_history

    # ---- Gradio ----
    if gradio:
        with gr.Blocks(title="Med-GRAG System") as demo:
            gr.Markdown("<h1>🧠 Med-GRAG 多轮对话系统</h1>")

            history_state = gr.State([])
            chatbot = gr.Chatbot(label="对话记录", height=450)
            msg = gr.Textbox(label="输入：")
            clear = gr.Button("清除")

            def respond(message, chat_history_list):
                final_answer, new_history = evaluate(message, chat_history_list)
                return new_history, "", new_history

            msg.submit(respond, [msg, history_state], [chatbot, msg, history_state], queue=False)
            clear.click(lambda: ([], []), None, [chatbot, history_state], queue=False)

        demo.launch(server_name="0.0.0.0", share=False)
    else:
        # 命令行测试
        history = []
        q1 = "我最近总是头痛，而且有点高血压，请问饮食上要注意什么？"
        print(f"\n======== Round 1 ========\nQ1: {q1}")
        a1, history = evaluate(q1, history)
        print(f"A1: {a1}")

        q2 = "那吃阿司匹林可以吗？"
        print(f"\n======== Round 2 ========\nQ2: {q2}")
        a2, history = evaluate(q2, history)
        print(f"A2: {a2}")


if __name__ == "__main__":
    fire.Fire(main)