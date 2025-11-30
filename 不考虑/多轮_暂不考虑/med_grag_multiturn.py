# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# med_grag_multiturn.py

# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
Med-GRAG Final: Chinese-LLaMA-2 + LoRA + Qwen-NER + Custom Medical Graph
适配用户自定义的 MedicalGraph 结构 (Disease, Drug, Food, Check...)
并实现多轮对话功能 (Multiturn)
"""
import os
import json
import fire
import gradio as gr
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from neo4j import GraphDatabase
# 导入 prompter
from utils.多轮.prompter_plus import Prompter

# ---------- 配置区域 ----------
# 阿里云百炼 API Key (请确保环境变量已设置或在此处手动填写)
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY",
                                            "sk-00459b72ffb245e5958c40c595d8ff67")  # ⚠️ 这是一个示例值，请替换为您的真实 Key

# Neo4j 配置 (使用你提供的账号密码)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "lty20001114"  # ⚠️ 这是一个示例值，请替换为您的真实密码

# 模型路径配置
# 请确保这些路径指向你本地的模型文件
BASE_MODEL_PATH = "chinese-llama-2-7b"
LORA_WEIGHTS_PATH = "lora-chinese-llama2-med/checkpoint-608"


# ---------- 1. 阿里云百炼 NER 模块 ----------
class AliyunNERExtractor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("⚠️ WARN: DASHSCOPE_API_KEY not found. Using local dictionary for NER.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # 简化的本地词典用于兜底
        self.medical_dict = {'感冒', '高血压', '糖尿病', '冠心病', '胃炎', '头痛', '发烧', '咳嗽', '阿司匹林'}

    def extract_entities(self, text):
        """提取实体"""
        if not self.api_key: return self._local_extract(text)
        try:
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system",
                     "content": "你是一个医学实体识别专家。从用户提供的文本中，提取所有可能的医学实体（疾病、症状、药品、检查、食物等）。只返回JSON格式的列表，如[\"感冒\", \"头痛\"]。"},
                    {"role": "user", "content": f"提取实体：{text}"}
                ],
                temperature=0.0
            )
            txt = completion.choices[0].message.content.strip()
            # 清理可能的 markdown 格式
            clean_text = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"❌ NER API Error, falling back to local dictionary: {e}")
            return self._local_extract(text)

    def _local_extract(self, text):
        # 简单匹配本地词典
        return [w for w in self.medical_dict if w in text]


# ---------- 2. 自定义图谱检索器 (适配你的 Schema) ----------
class MedicalGraphRetriever:
    def __init__(self, uri, user, pwd):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self.driver.verify_connectivity()
            print("✅ Neo4j connected successfully.")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            self.driver = None

    def query_entity_context(self, entity_name):
        """
        针对图谱结构设计的全方位查询。
        """
        if not self.driver: return ""

        context_parts = []

        # 定义关系映射表
        type_map = {
            'recommand_eat': '推荐食谱', 'no_eat': '忌吃食物', 'do_eat': '宜吃食物',
            'common_drug': '常用药品', 'recommand_drug': '推荐药品',
            'need_check': '所需检查', 'has_symptom': '典型症状',
            'acompany_with': '并发症', 'belongs_to': '所属科室'
        }

        with self.driver.session() as session:
            # 1. 查询是否是【疾病】节点 (Disease)
            # 获取：简介、成因、预防、治疗方式
            q_disease_info = """
            MATCH (n:Disease {name: $name})
            RETURN n.desc AS desc, n.prevent AS prevent, n.cause AS cause, 
                   n.easy_get AS easy_get, n.cure_way AS cure_way
            """
            result = session.run(q_disease_info, name=entity_name).single()

            if result:
                info = result
                context_parts.append(f"【疾病：{entity_name}的基本信息】")
                if info['desc']: context_parts.append(f"简介：{info['desc']}")
                if info['cause']: context_parts.append(f"成因：{info['cause']}")
                if info['prevent']: context_parts.append(f"预防：{info['prevent']}")
                if info['cure_way']: context_parts.append(f"治疗方式：{info['cure_way']}")

                # 查询疾病的【关联关系】
                q_rels = """
                MATCH (n:Disease {name: $name})-[r]->(m)
                RETURN type(r) AS type, m.name AS target
                """
                rels = session.run(q_rels, name=entity_name)

                rel_dict = {}
                for r in rels:
                    t = type_map.get(r['type'], r['type'])
                    if t not in rel_dict: rel_dict[t] = []
                    rel_dict[t].append(r['target'])

                for k, v in rel_dict.items():
                    context_parts.append(f"{k}：{'、'.join(v[:10])}")

            # 3. 查询是否是【症状】节点 (Symptom) -> 查可能患有的疾病
            q_symptom = """
            MATCH (n:Disease)-[:has_symptom]->(s {name: $name})
            RETURN n.name AS disease
            LIMIT 10
            """
            res_sym = session.run(q_symptom, name=entity_name)
            diseases = [r['disease'] for r in res_sym]
            if diseases:
                context_parts.append(f"【症状：{entity_name}】可能是以下疾病的症状：{'、'.join(diseases)}")

            # 4. 查询是否是【药品】节点 (Drug) -> 查主治疾病
            q_drug = """
            MATCH (d:Disease)-[r]-(dr)
            WHERE type(r) IN ['common_drug', 'recommand_drug'] AND dr.name = $name
            RETURN d.name AS disease
            LIMIT 10
            """
            res_drug = session.run(q_drug, name=entity_name)
            treated = [r['disease'] for r in res_drug]
            if treated:
                context_parts.append(f"【药品：{entity_name}】常用于治疗：{'、'.join(treated)}")

        return "\n".join(context_parts)


# 初始化组件
ner_extractor = AliyunNERExtractor()
kg_retriever = MedicalGraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PWD)

# ---------- 3. 模型推理主逻辑 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(
        load_8bit: bool = False,
        base_model: str = BASE_MODEL_PATH,
        use_lora: bool = True,
        lora_weights: str = LORA_WEIGHTS_PATH,
        prompt_template: str = "med_template",
        gradio: bool = False,
):
    # ---- 模型加载 ----
    prompter = Prompter(prompt_template)

    # 尝试加载 LoRA 模型
    try:
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
        print("✅ LLM loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading LLM/LoRA weights: {e}")

        # 在无法加载模型时，使用一个哑函数（Mock Function）进行测试
        def mock_evaluate(instruction, history, **kwargs):
            formatted_history = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
            print(f"--- Mock Evaluation ---\nHistory:\n{formatted_history}\nQuestion: {instruction}")

            # 使用 NER/KG 检索结果进行简单的模拟回复
            full_context = f"{formatted_history}\n用户：{instruction}"
            entities = ner_extractor.extract_entities(full_context)
            kg_context = ""
            if entities:
                kg_context = kg_retriever.query_entity_context(entities[0])  # 只查第一个

            if kg_context:
                mock_answer = f"【模拟RAG回答】根据您提到的'{entities[0]}'，我们从知识库中检索到以下信息：\n{kg_context.splitlines()[0]}...\n请问您还有其他症状吗？"
            else:
                mock_answer = f"【模拟回答】您提到了'{instruction}'。我没有找到相关知识。请您详细描述一下您最近感觉如何？"

            new_history = history + [(instruction, mock_answer)]
            return mock_answer, new_history

        evaluate = mock_evaluate
        print("⚠️ Falling back to Mock Evaluation. Cannot run full LLM RAG without model weights.")

    # 格式化历史记录，作为 LLM 的 Context
    def format_history(history):
        formatted = []
        for user_q, bot_a in history:
            formatted.append(f"用户：{user_q}")
            formatted.append(f"助手：{bot_a}")
        return "\n".join(formatted)

    @torch.no_grad()
    def evaluate(instruction, history, **kwargs):  # 👈 多轮对话入口

        current_question = instruction.strip()
        formatted_history = format_history(history)

        # 将历史对话和当前问题拼接，作为完整的上下文，用于 NER 提取
        full_context_text = f"{formatted_history}\n用户：{current_question}"

        # 1. 实体提取 (对完整的上下文进行提取)
        entities = ner_extractor.extract_entities(full_context_text)
        print(f"🔍 [NER]: {entities}")

        # 2. 知识图谱检索 (RAG)
        kg_context = ""
        if entities:
            contexts = []
            for ent in entities:
                # 遍历所有实体进行检索
                info = kg_retriever.query_entity_context(ent)
                if info: contexts.append(info)
            kg_context = "\n\n".join(contexts)

        print(f"📚 [Graph Context]:\n{kg_context[:200]}..." if kg_context else "📚 [Graph Context]: None")

        # 3. 构造 Prompt

        # 引导性指令：指导 LLM 优先参考知识库和历史，并进行追问
        guiding_instruction = (
            "你是一位专业的医疗助手，请优先参考提供的【知识库信息】和【历史对话】来回答当前用户的问题。 "
            "如果【知识库信息】不足以得出结论，或者用户的描述像是一个新病症，请**礼貌地进行追问**，例如询问更具体的症状、持续时间、发作频率或近期活动，以提供更准确的指导。"
        )

        input_context = f"【知识库信息】:\n{kg_context or '无相关信息'}\n\n【历史对话】:\n{formatted_history or '首次对话'}"

        prompt = prompter.generate_prompt(
            instruction=guiding_instruction,
            # 将当前用户问题作为 Input 的一部分
            input=f"当前用户问题：{current_question}\n\n{input_context}"
        )

        # 4. 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)

        output_ids = model.generate(
            **inputs,
            generation_config=GenerationConfig(
                temperature=0.1, top_p=0.75, top_k=40, num_beams=1, max_new_tokens=512
            )
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        final_answer = prompter.get_response(response)

        # 5. 更新历史记录
        new_history = history + [(current_question, final_answer)]

        return final_answer, new_history

    # ---- 启动 Gradio 界面 ----
    if gradio:
        with gr.Blocks(title="Med-GRAG System") as demo:
            gr.Markdown("<h1>🧠 Med-GRAG 多轮对话系统</h1>")
            gr.Markdown("Chinese-LLaMA-2 + LoRA + Qwen-NER + Custom Medical Graph (支持多轮上下文理解)")

            # gr.State 存储历史对话 [(用户问, 助手答), ...]
            history_state = gr.State([])

            chatbot = gr.Chatbot(label="🩺 对话记录", height=400)
            msg = gr.Textbox(label="输入你的问题或症状描述：")

            clear = gr.Button("🗑️ 清除历史记录")

            def respond(message, chat_history_list):
                # 调用 evaluate，返回 最终回答 和 更新后的历史列表
                final_answer, new_history = evaluate(message, chat_history_list)

                # 返回给 Gradio 组件
                return new_history, "", new_history

            # 绑定事件
            msg.submit(
                respond,
                [msg, history_state],
                [chatbot, msg, history_state],
                queue=False
            )

            # 清除历史记录事件：清空 Chatbot 和 State
            clear.click(lambda: ([], []), None, [chatbot, history_state], queue=False)

        demo.launch(server_name="0.0.0.0", share=False)
    else:
        # 非 Gradio 模式的简单多轮测试
        history = []
        q1 = "我最近总是头痛，而且有点高血压，请问饮食上要注意什么？"
        print(f"\n======== 第一次对话 ========\nQ1: {q1}")
        a1, history = evaluate(q1, history)
        print(f"A1: {a1}")

        q2 = "那吃阿司匹林可以吗？这个药有什么作用？"
        print(f"\n======== 第二次对话 (RAG 引用 Q1 的实体) ========\nQ2: {q2}")
        a2, history = evaluate(q2, history)
        print(f"A2: {a2}")


if __name__ == "__main__":
    fire.Fire(main)