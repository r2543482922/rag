import json
import sys
from collections import defaultdict

import faiss
import fire
import gradio as gr
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

##############################
# 第一部分：构建知识检索库
##############################

# 定义 MRCONSO.RRF 文件列名
columns_conso = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI',
                 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF']


# 读取 MRCONSO 文件（概念和名称）
def read_mrconso(file_path):
    try:
        mrconso = pd.read_csv(file_path, sep='|', names=columns_conso, dtype=str, encoding='utf-8')
        print("MRCONSO 文件加载成功。")
        return mrconso
    except Exception as e:
        print(f"加载 MRCONSO 文件出错: {e}")
        return None


# 定义 MRREL.RRF 文件列名
columns_rel = ['CUI1', 'CUI2', 'REL', 'RELA', 'RUI', 'SAB', 'SL', 'SUPPRESS']


# 读取 MRREL 文件（概念之间的关系）
def read_mrrel(file_path):
    try:
        mrrel = pd.read_csv(file_path, sep='|', names=columns_rel, dtype=str, encoding='utf-8')
        print("MRREL 文件加载成功。")
        return mrrel
    except Exception as e:
        print(f"加载 MRREL 文件出错: {e}")
        return None


# 请将以下路径替换为你实际的 UMLS 文件路径
MRCONSO_PATH = 'path_to_MRCONSO.RRF'
MRREL_PATH = 'path_to_MRREL.RRF'

# 加载 UMLS 数据
mrconso = read_mrconso(MRCONSO_PATH)
mrrel = read_mrrel(MRREL_PATH)

# 创建概念字典：CUI -> [名称1, 名称2, ...]
concept_dict = defaultdict(list)
if mrconso is not None:
    for _, row in mrconso.iterrows():
        cui = row['CUI']
        term = row['STR']
        concept_dict[cui].append(term)


# 将 MRREL 中的关系转为自然语言句子
def convert_relationship_to_text(mrrel, concept_dict):
    sentences = []
    for _, row in mrrel.iterrows():
        cui1 = row['CUI1']
        cui2 = row['CUI2']
        relation = row['REL']  # 关系类型，例如 "associated_with"
        terms1 = concept_dict.get(cui1, ["Unknown"])
        terms2 = concept_dict.get(cui2, ["Unknown"])
        for term1 in terms1:
            for term2 in terms2:
                sentence = f"{term1} {relation.replace('_', ' ')} {term2}."
                sentences.append(sentence)
    return sentences


# 转换后得到自然语言描述的关系句子列表
knowledge_sentences = convert_relationship_to_text(mrrel, concept_dict)
print("示例关系句子:", knowledge_sentences[:5])

# 构建语义检索库
# 这里我们使用一个多语言的 SentenceTransformer 模型，可根据需求选择医学专用模型
retrieval_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
knowledge_embeddings = retrieval_model.encode(knowledge_sentences, convert_to_numpy=True)

# 使用 FAISS 构建索引
faiss_index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
faiss_index.add(knowledge_embeddings)


# 检索函数：根据查询返回 top_k 个相关知识句子
def retrieve_knowledge(query, top_k=3):
    q_emb = retrieval_model.encode([query])
    D, I = faiss_index.search(np.array(q_emb), top_k)
    return [knowledge_sentences[i] for i in I[0]]


##############################
# 第二部分：模型推理代码（基于 Huatuo + 知识增强）
##############################

# 加载测试数据（这里为批量引入问题）
def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            d = json.loads(line)
            input_data.append(d)
    return input_data


def main(
        load_8bit: bool = False,
        base_model: str = "llama-7b",
        instruct_dir: str = "data/infer.json",  # 推理数据集路径
        use_lora: bool = True,
        lora_weights: str = "lora-llama-med",
        prompt_template: str = "med_template",
):
    # 初始化 Prompt 模板
    prompter = Prompter(prompt_template)
    # 加载 tokenizer 和基模型
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_lora:
        print(f"Using LoRA weights from {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # 使用 0 作为 pad token
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()  # 模型精度转换
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # 定义推理函数，加入知识增强
    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        # 先从知识库检索相关知识
        retrieved_context = retrieve_knowledge(instruction, top_k=3)
        # 生成原始 prompt（使用用户自定义模板）
        base_prompt = prompter.generate_prompt(instruction, input)
        # 拼接检索到的知识与原始 prompt
        prompt = f"以下是相关医学知识：{'; '.join(retrieved_context)}\n\n" + base_prompt

        inputs_tok = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs_tok["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return prompter.get_response(output)

    def infer_from_json(instruct_dir):
        input_data = load_instruction(instruct_dir)
        for d in input_data:
            instruction = d["instruction"]
            output = d["output"]
            print("### Inference ###")
            model_output = evaluate(instruction)
            print("### Instruction ###")
            print(instruction)
            print("### Golden Output ###")
            print(output)
            print("### Model Output ###")
            print(model_output)
            print("====================================")

    if instruct_dir != "":
        infer_from_json(instruct_dir)
    else:
        for instruction in [
            "我感冒了，怎么治疗",
            "一个患有肝衰竭综合征的病人，除了常见的临床表现外，还有哪些特殊的体征？",
            "急性阑尾炎和缺血性心脏病的多发群体有何不同？",
            "小李最近出现了心动过速的症状，伴有轻度胸痛。体检发现P-R间期延长，伴有T波低平和ST段异常",
        ]:
            print("Instruction:", instruction)
            print("Response:", evaluate(instruction))
            print()


if __name__ == "__main__":
    fire.Fire(main)
