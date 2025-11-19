#!/usr/bin/env python3
"""
Chinese-LLaMA-2-7B + LoRA + Knowledge Graph 医学推理脚本
"""
import sys
import os
import json
import re
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from neo4j import GraphDatabase

from utils.prompter import Prompter

# ---------- 环境变量配置 ----------
# 请在系统环境变量中设置这些值，或取消注释下面几行：
# os.environ["DEEPSEEK_API_KEY"] = "sk-your-free-key"
# os.environ["NEO4J_URI"] = "bolt://localhost:7687"
# os.environ["NEO4J_USER"] = "neo4j"
# os.environ["NEO4J_PWD"] = "password"

# ---------- 通用大模型实体识别 ----------
try:
    NER_CLIENT = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    HAS_NER = True
except Exception:
    print("Warning: DeepSeek API not available, entity recognition disabled")
    HAS_NER = False


def extract_entities(text: str) -> list[str]:
    """零样本抽取医学实体，返回 List[str]"""
    if not HAS_NER:
        # 降级策略：简单的中文实体提取
        return re.findall(r"[\u4e00-\u9fff]{2,10}", text)

    sys_prompt = ("从句子中抽取出所有医学实体（疾病、药品、检查、手术、症状），"
                  "以 JSON 列表返回，如 [\"实体1\", \"实体2\"]")
    try:
        rsp = NER_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=128
        )
        return json.loads(rsp.choices[0].message.content)
    except Exception as e:
        print(f"Entity extraction failed: {e}")
        return re.findall(r"[\u4e00-\u9fff]{2,10}", text)


# ---------- Neo4j 子图召回 ----------
try:
    NEO4J_DRIVER = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PWD", "password")
        )
    )
    HAS_KG = True
except Exception as e:
    print(f"Warning: Neo4j connection failed: {e}")
    HAS_KG = False
    NEO4J_DRIVER = None


def retrieve_subgraph(entities: list[str], max_hops: int = 2, top_k: int = 10) -> str:
    """返回人类可读子图文本"""
    if not HAS_KG or not entities:
        return ""

    try:
        with NEO4J_DRIVER.session() as sess:
            result = sess.run(
                f"""
                MATCH path=(e:Entity)-[*1..{max_hops}]-(rel:Entity)
                WHERE e.name IN $ents
                RETURN DISTINCT e.name + ' ' + type(relationships(path)[0]) + ' ' + rel.name AS triple
                LIMIT {top_k}
                """,
                ents=entities
            )
            triples = [r["triple"] for r in result]
        return "；".join(triples) if triples else ""
    except Exception as e:
        print(f"Knowledge graph query failed: {e}")
        return ""


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_instruction(instruct_dir):
    """加载测试指令数据"""
    if not os.path.exists(instruct_dir):
        return []

    input_data = []
    with open(instruct_dir, "r", encoding="utf-8-sig") as f:
        for line in f:
            try:
                input_data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return input_data


def main(
        load_8bit: bool = False,
        base_model: str = "hfl/chinese-llama-2-7b",
        instruct_dir: str = "data/infer.json",
        use_lora: bool = True,
        lora_weights: str = "lora-chinese-llama2-med/checkpoint-608",
        prompt_template: str = "med_template",
        gradio: bool = False,
):
    # ---------- 模型加载 ----------
    prompter = Prompter(prompt_template)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if use_lora and os.path.exists(lora_weights):
        print(f"Loading LoRA weights: {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)

    # 配置tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    @torch.no_grad()
    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            do_sample=True,
            max_new_tokens=512,
            **kwargs,
    ):
        # ---- 知识图谱增强 ----
        question = f"{instruction} {input or ''}".strip()
        entities = extract_entities(question)
        print(f"Extracted entities: {entities}")

        kg_text = retrieve_subgraph(entities)
        if kg_text:
            print(f"Retrieved knowledge: {kg_text}")
            input = f"相关知识：{kg_text}\n\n问题：{input or ''}".strip()

        # ---- LLM生成 ----
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)

        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        gen_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

        output_ids = model.generate(
            **inputs,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return prompter.get_response(response)

    # ---------- 命令行模式 ----------
    if not gradio:
        if instruct_dir and os.path.exists(instruct_dir):
            data = load_instruction(instruct_dir)
            for i, d in enumerate(data):
                instruction, golden = d.get("instruction", ""), d.get("output", "")
                print(f"\n--- Sample {i + 1} ---")
                print("### Instruction ###")
                print(instruction)
                print("### Golden Answer ###")
                print(golden)
                print("### Model Response ###")
                response = evaluate(instruction)
                print(response)
                print("-" * 60)
        else:
            # 默认测试问题
            test_questions = [
                "我感冒了，怎么治疗？",
                "肝衰竭有哪些特殊体征？",
                "急性阑尾炎和缺血性心脏病的好发人群有何不同？",
            ]
            for question in test_questions:
                print(f"\nInstruction: {question}")
                response = evaluate(question)
                print(f"Response: {response}")
        return

    # ---------- Gradio Web界面 ----------
    demo = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.Textbox(label="Instruction", placeholder="请输入医学问题…"),
            gr.Textbox(label="Input (可选)", placeholder="额外上下文…"),
            gr.Slider(0, 1, 0.1, label="Temperature"),
            gr.Slider(1, 50, 40, step=1, label="Top-k"),
            gr.Slider(0, 1, 0.75, label="Top-p"),
            gr.Slider(1, 8, 1, step=1, label="Beams"),  # 默认改为1，与代码一致
            gr.Slider(64, 1024, 512, step=64, label="Max new tokens"),
        ],
        outputs=gr.Textbox(label="Response", lines=10),
        title="Chinese-LLaMA-2-7B + LoRA + 知识图谱 医学问答",
        description="输入医学问题，系统会从知识图谱检索相关信息并生成回答"
    )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    fire.Fire(main)