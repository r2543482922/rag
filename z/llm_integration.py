# -*- coding: utf-8 -*-
import fire
import gradio as gr
import torch
from peft import PeftModel
from py2neo import Graph
from question_classifier import QuestionClassifier
from question_parser import QuestionPaser
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"


class ConversationalMedicalAssistant:
    def __init__(self, llm_config, kg_config):
        # 初始化大模型
        self.initialize_llm(llm_config)

        # 初始化知识图谱
        self.kg_searcher = KnowledgeGraphSearcher(
            uri=kg_config['uri'],
            auth=(kg_config['user'], kg_config['password'])
        )

        # 初始化分类器和解析器
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()

        # 对话历史
        self.conversation_history = []
        self.max_history = 5

    def initialize_llm(self, config):
        """初始化大语言模型"""
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            config['base_model'],
            load_in_8bit=config['load_8bit'],
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if config['use_lora']:
            print(f"Using LoRA weights: {config['lora_weights']}")
            self.llm_model = PeftModel.from_pretrained(
                self.llm_model,
                config['lora_weights'],
                torch_dtype=torch.float16,
            )

        self.llm_model.config.pad_token_id = self.llm_tokenizer.pad_token_id = 0
        self.llm_model.config.bos_token_id = 1
        self.llm_model.config.eos_token_id = 2

        if not config['load_8bit']:
            self.llm_model.half()

        self.llm_model.eval()
        self.prompter = Prompter(config['prompt_template'])

    def get_kg_information(self, question):
        """从知识图谱获取结构化信息"""
        try:
            # 分类问题
            res_classify = self.classifier.classify(question)
            if not res_classify:
                return None

            # 生成查询
            res_sql = self.parser.parser_main(res_classify)
            if not res_sql:
                return None

            # 执行查询
            return self.kg_searcher.search_main(res_sql)
        except Exception as e:
            print(f"知识图谱查询错误: {str(e)}")
            return None

    def generate_conversational_response(self, user_input):
        """生成对话式响应"""
        # 获取知识图谱信息
        kg_info = self.get_kg_information(user_input)

        # 更新对话历史
        self.update_conversation_history(f"用户: {user_input}")

        # 构建提示
        prompt = self.build_conversation_prompt(user_input, kg_info)

        # 生成响应
        response = self.generate_with_llm(prompt)

        # 更新对话历史
        self.update_conversation_history(f"助手: {response}")

        return response

    def build_conversation_prompt(self, user_input, kg_info=None):
        """构建对话提示"""
        # 基础提示
        prompt = f"你是一位专业、友善的医疗助手。请根据以下信息回答用户问题。\n"

        # 添加对话历史
        if self.conversation_history:
            prompt += "\n对话历史:\n" + "\n".join(self.conversation_history[-self.max_history:]) + "\n"

        # 添加知识图谱信息
        if kg_info and kg_info['status'] == 'success':
            prompt += "\n医疗知识库信息:\n"
            for detail in kg_info['data']['details']:
                if detail['question_type'] == 'disease_symptom':
                    diseases = {r['m.name'] for r in detail['raw_results']}
                    symptoms = {r['n.name'] for r in detail['raw_results']}
                    prompt += f"- {', '.join(diseases)}的症状包括: {', '.join(symptoms)}\n"
                elif detail['question_type'] == 'disease_desc':
                    for record in detail['raw_results']:
                        prompt += f"- {record['m.name']}: {record['m.desc']}\n"
                # 可以添加更多问题类型的处理...

        # 添加当前问题
        prompt += f"\n当前问题: {user_input}\n请提供专业、清晰且友好的回答:"

        return prompt

    def generate_with_llm(self, prompt):
        """使用大模型生成响应"""
        inputs = self.llm_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_beams=3,
            do_sample=True,
        )

        with torch.no_grad():
            generation_output = self.llm_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=512,
            )

        output = self.llm_tokenizer.decode(generation_output.sequences[0])
        return self.clean_response(output)

    def clean_response(self, text):
        """清理模型输出"""
        # 移除提示部分
        if "助手:" in text:
            text = text.split("助手:")[-1]

        # 移除特殊标记
        for marker in ["<|endoftext|>", "[INST]", "[/INST]"]:
            text = text.replace(marker, "")

        return text.strip()

    def update_conversation_history(self, text):
        """更新对话历史"""
        self.conversation_history.append(text)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)


class KnowledgeGraphSearcher:
    """知识图谱查询类（与前文相同）"""

    def __init__(self, uri, auth):
        try:
            self.g = Graph(uri, auth=auth, secure=False)
            self.num_limit = 20
            self.g.run("RETURN 1").data()
            print("Neo4j连接成功")
        except Exception as e:
            print(f"Neo4j连接失败: {str(e)}")
            raise

    def search_main(self, sqls):
        response = {
            "status": "success",
            "data": {
                "friendly_answer": "",
                "details": []
            },
            "metadata": {
                "query_count": len(sqls),
                "result_count": 0
            }
        }

        final_answers = []

        for sql_ in sqls:
            question_type = sql_['question_type']
            queries = sql_['sql']

            for query in queries:
                detail_item = {
                    "question_type": question_type,
                    "cypher_query": query,
                    "nodes": [],
                    "relationships": [],
                    "raw_results": []
                }

                try:
                    ress = self.g.run(query).data()
                    response["metadata"]["result_count"] += len(ress)
                    detail_item["raw_results"] = ress

                    for record in ress:
                        for key, value in record.items():
                            if key.startswith('m.') or key.startswith('n.'):
                                node_name = value
                                node_type = 'Disease' if 'm.name' in key else key.split('.')[0].upper()
                                node_info = {
                                    "identifier": key,
                                    "name": node_name,
                                    "label": node_type
                                }
                                detail_item["nodes"].append(node_info)

                            elif key == 'r.name':
                                rel_info = {
                                    "type": value,
                                    "from": record.get('m.name', ''),
                                    "to": record.get('n.name', '')
                                }
                                detail_item["relationships"].append(rel_info)

                    response["data"]["details"].append(detail_item)
                    answers = self.answer_prettify(question_type, ress)
                    if answers:
                        final_answers.append(answers)

                except Exception as e:
                    error_item = {
                        "question_type": question_type,
                        "cypher_query": query,
                        "error": str(e)
                    }
                    response["data"]["details"].append(error_item)
                    response["status"] = "partial_success"

        response["data"]["friendly_answer"] = '\n'.join(final_answers) if final_answers else '未找到相关信息'
        return response

    def answer_prettify(self, question_type, answers):
        if not answers:
            return ''

        if question_type == 'disease_symptom':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            return f'{subject}的症状包括：{"；".join(list(set(desc))[:self.num_limit])}'

        elif question_type == 'disease_desc':
            desc = [i['m.desc'] for i in answers]
            subject = answers[0]['m.name']
            return f'{subject},熟悉一下：{"；".join(list(set(desc))[:self.num_limit])}'

        return ''


def main(
        load_8bit: bool = False,
        base_model: str = "llama-7b",
        use_lora: bool = True,
        lora_weights: str = "lora-llama-med",
        prompt_template: str = "med_template",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "lty20001114",
        launch_gradio: bool = True
):
    # 初始化配置
    llm_config = {
        'base_model': base_model,
        'load_8bit': load_8bit,
        'use_lora': use_lora,
        'lora_weights': lora_weights,
        'prompt_template': prompt_template
    }

    kg_config = {
        'uri': neo4j_uri,
        'user': neo4j_user,
        'password': neo4j_password
    }

    # 初始化对话助手
    assistant = ConversationalMedicalAssistant(llm_config, kg_config)

    if launch_gradio:
        # 创建Gradio界面
        def chat_interface(message, history):
            response = assistant.generate_conversational_response(message)
            return response

        iface = gr.ChatInterface(
            fn=chat_interface,
            title="医疗智能对话助手",
            description="我是您的专业医疗助手，可以回答各种医疗健康相关问题。",
            examples=[
                "感冒了应该怎么办？",
                "糖尿病的典型症状有哪些？",
                "高血压患者饮食上需要注意什么？"
            ],
            theme="soft",
            cache_examples=True
        )
        iface.launch(share=True)
    else:
        # 命令行交互模式
        print("医疗智能对话助手已启动，输入'退出'结束对话")
        print("=====================================")

        while True:
            try:
                user_input = input("您: ").strip()
                if user_input.lower() in ['退出', 'exit', 'quit']:
                    print("助手: 再见！祝您健康！")
                    break

                if not user_input:
                    continue

                response = assistant.generate_conversational_response(user_input)
                print("\n助手:", response)
                print()

            except KeyboardInterrupt:
                print("\n助手: 再见！祝您健康！")
                break
            except Exception as e:
                print(f"\n助手: 抱歉，我遇到了些问题。请稍后再试。错误: {str(e)}")
                print()


if __name__ == "__main__":
    fire.Fire(main)
