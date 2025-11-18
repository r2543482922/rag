#!/usr/bin/env python3
"""
Chinese-LLaMA-2-7B + LoRA + RAG 医学推理脚本
用法：
    # 命令行
    python infer_lora_med.py \
        --base_model hfl/chinese-llama-2-7b \
        --lora_weights ./lora-chinese-llama2-med-e5 \
        --rag

    # Gradio
    python infer_lora_med.py --gradio --rag
"""
import sys
import os
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

# ---------- 可选 RAG ----------
USE_RAG = False  # 默认关闭，命令行 --rag 打开
if USE_RAG:
    from retriever import MedRetriever

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r", encoding="utf-8-sig") as f:
        for line in f:
            input_data.append(json.loads(line.strip()))
    return input_data


def main(
        load_8bit: bool = False,
        base_model: str = "chinese-llama-2-7b",
        instruct_dir: str = "data/infer.json",  # 留空则跑内置问题
        use_lora: bool = True,
        lora_weights: str = "lora-chinese-llama2-med/checkpoint-608",
        prompt_template: str = "med_template",
        rag: bool = False,
        gradio: bool = False,
):
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_lora:
        print(f"Using LoRA weights: {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # ---------- RAG ----------
    retriever = MedRetriever(index_path="med.index") if rag else None

    @torch.no_grad()
    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,  # 关闭 beam
            do_sample=True,
            max_new_tokens=512,
            **kwargs,
    ):
        knowledge = retriever.retrieve(instruction) if retriever else ""
        if knowledge:
            input = (knowledge + "\n\n" + (input or "")).strip()
        prompt = prompter.generate_prompt(instruction, input)

        # 1. 编码并去掉 token_type_ids
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        # 2. 生成
        gen_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        output_ids = model.generate(
            **inputs,
            generation_config=gen_config,
            max_new_tokens=max_new_tokens,
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return prompter.get_response(response)

    # ---------- 1. 命令行 ----------
    if not gradio:
        if instruct_dir and os.path.exists(instruct_dir):
            data = load_instruction(instruct_dir)
            for d in data:
                instruction, golden = d["instruction"], d["output"]
                print("### Instruction ###")
                print(instruction)
                print("### Golden ###")
                print(golden)
                print("### Model ###")
                print(evaluate(instruction))
                print("-" * 60)
        else:
            for instruction in [
                "我感冒了，怎么治疗？",
                "肝衰竭有哪些特殊体征？",
                "急性阑尾炎和缺血性心脏病的好发人群有何不同？",
            ]:
                print("Instruction:", instruction)
                print("Response:", evaluate(instruction))
                print()
        return

    # ---------- 2. Gradio ----------
    demo = gr.Interface(
        fn=evaluate,
        inputs=[
            gr.Textbox(label="Instruction", placeholder="请输入医学问题…"),
            gr.Textbox(label="Input (可选)", placeholder="额外上下文…"),
            gr.Slider(0, 1, 0.1, label="Temperature"),
            gr.Slider(1, 50, 40, step=1, label="Top-k"),
            gr.Slider(0, 1, 0.75, label="Top-p"),
            gr.Slider(1, 8, 4, step=1, label="Beams"),
            gr.Slider(64, 1024, 512, step=64, label="Max new tokens"),
        ],
        outputs=gr.Textbox(label="Response", lines=10),
        title="Chinese-LLaMA-2-7B + LoRA + RAG 医学问答",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    fire.Fire(main)