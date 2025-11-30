#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Med-GRAG Final: Chinese-LLaMA-2 + LoRA + Qwen-NER + Custom Medical Graph
适配用户自定义的 MedicalGraph 结构 (Disease, Drug, Food, Check...)
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
from utils.prompter import Prompter

# ---------- 配置区域 ----------
# 阿里云百炼 API Key
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "sk-00459b72ffb245e5958c40c595d8ff67")

# Neo4j 配置 (使用你提供的账号密码)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "lty20001114"  # 你的密码


# ---------- 1. 阿里云百炼 NER 模块 ----------
class AliyunNERExtractor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # 简化的本地词典用于兜底
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
            txt = completion.choices, [object Object], message.content.strip()
            clean_text = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"NER API Error: {e}")
            return self._local_extract(text)

    def _local_extract(self, text):
        return [w for w in self.medical_dict if w in text]


# ---------- 2. 自定义图谱检索器 (适配你的 Schema) ----------
class MedicalGraphRetriever:
    def __init__(self, uri, user, pwd):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            print("✅ Neo4j connected successfully.")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            self.driver = None

    def query_entity_context(self, entity_name):
        """
        针对你的图谱结构设计的全方位查询。
        如果实体是'Disease'，查询它的症状、药物、忌口等。
        如果实体是'Symptom'，查询可能对应的疾病。
        """
        if not self.driver: return ""

        context_parts = []

        with self.driver.session() as session:
            # 1. 查询是否是【疾病】节点 (Disease)
            # 获取：简介、预防、易感人群、治愈率、科室
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
                if info['prevent']: context_parts.append(f"预防：{info['prevent']}")
                if info['cure_way']: context_parts.append(f"治疗方式：{info['cure_way']}")

                # 2. 查询疾病的【关联关系】 (根据你的 create_graphrels 定义)
                # 推荐吃(recommand_eat), 忌吃(no_eat), 宜吃(do_eat),
                # 常用药(common_drug), 好评药(recommand_drug),
                # 检查(need_check), 症状(has_symptom), 并发症(acompany_with)

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
                    context_parts.append(f"{k}：{'、'.join(v[:10])}")  # 限制数量防止Prompt过长

            # 3. 查询是否是【症状】节点 (Symptom) -> 查可能患有的疾病
            q_symptom = """
            MATCH (n:Disease)-[:has_symptom]->(s:Symptom {name: $name})
            RETURN n.name AS disease
            LIMIT 10
            """
            res_sym = session.run(q_symptom, name=entity_name)
            diseases = [r['disease'] for r in res_sym]
            if diseases:
                context_parts.append(f"【{entity_name}】可能是以下疾病的症状：{'、'.join(diseases)}")

            # 4. 查询是否是【药品】节点 (Drug) -> 查主治疾病
            q_drug = """
            MATCH (d:Disease)-[:common_drug|recommand_drug]-(dr:Drug {name: $name})
            RETURN d.name AS disease
            LIMIT 10
            """
            res_drug = session.run(q_drug, name=entity_name)
            treated = [r['disease'] for r in res_drug]
            if treated:
                context_parts.append(f"【{entity_name}】常用于治疗：{'、'.join(treated)}")

        return "\n".join(context_parts)


# 初始化组件
ner_extractor = AliyunNERExtractor()
kg_retriever = MedicalGraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PWD)

# ---------- 3. 模型推理主逻辑 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(
        load_8bit: bool = False,
        base_model: str = "hfl/chinese-llama-2-7b",
        use_lora: bool = True,
        lora_weights: str = "lora-chinese-llama2-med/checkpoint-608",
        prompt_template: str = "med_template",
        gradio: bool = False,
):
    # ---- 模型加载 ----
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map="auto",
    )
    if use_lora:
        model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit: model.half().eval()

    @torch.no_grad()
    def evaluate(instruction, input=None, **kwargs):
        # 1. 实体提取
        question = f"{instruction} {input or ''}".strip()
        entities = ner_extractor.extract_entities(question)
        print(f"🔍 [NER]: {entities}")

        # 2. 知识图谱检索 (RAG)
        kg_context = ""
        if entities:
            # 对每个实体进行检索，拼接结果
            contexts = []
            for ent in entities:
                info = kg_retriever.query_entity_context(ent)
                if info: contexts.append(info)
            kg_context = "\n\n".join(contexts)

        print(f"📚 [Graph Context]:\n{kg_context[:200]}..." if kg_context else "📚 [Graph Context]: None")

        # 3. 构造 Prompt
        # 核心：将图谱知识作为 Context 注入
        if kg_context:
            input_context = (
                f"以下是检索到的医学知识库信息，请优先参考这些信息回答用户问题：\n"
                f"---------------------\n"
                f"{kg_context}\n"
                f"---------------------\n"
                f"用户补充信息：{input or '无'}"
            )
        else:
            input_context = input or ""

        prompt = prompter.generate_prompt(instruction, input_context)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 4. 生成
        output_ids = model.generate(
            **inputs,
            generation_config=GenerationConfig(
                temperature=0.1, top_p=0.75, top_k=40, num_beams=1, max_new_tokens=512
            )
        )
        response = tokenizer.decode(output_ids, [object Object],, skip_special_tokens = True)
        return prompter.get_response(response)

    # ---- 启动界面 ----
    if gradio:
        gr.Interface(
            fn=evaluate,
            inputs=[gr.Textbox(label="问题"), gr.Textbox(label="补充信息")],
            outputs=gr.Textbox(label="回答"),
            title="Med-GRAG System",
            description="Chinese-LLaMA-2 + LoRA + Qwen-NER + Custom Medical Graph"
        ).launch(server_name="0.0.0.0", share=False)
    else:
        # 测试用例
        q = "我最近总是头痛，而且有点高血压，请问饮食上要注意什么？"
        print(f"\nQuestion: {q}")
        print(f"Answer: {evaluate(q)}")


if __name__ == "__main__":
    fire.Fire(main)