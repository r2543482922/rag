# utils/prompter.py

import json
import os

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # 使用默认 Alpaca/LLaMA 模板
            self.template = {
                "description": "Template for medical RAG with history",
                "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
                "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
                "response_split": "### Response:",
            }
        else:
            # 假设你的med_template文件包含在本地（这里为了简洁，直接硬编码）
            # 实际应用中，你可能需要加载一个json文件
            if template_name == "med_template":
                self.template = {
                    "description": "Medical RAG template with instruction, context (input) and response.",
                    "prompt_input": "你是一位专业的医疗助手。请根据下面提供的指令和补充信息（包括知识库和历史对话）给出完整、准确的回答。\n\n### 指令：\n{instruction}\n\n### 补充信息：\n{input}\n\n### 回答：\n",
                    "prompt_no_input": "你是一位专业的医疗助手。请根据下面提供的指令给出完整、准确的回答。\n\n### 指令：\n{instruction}\n\n### 回答：\n",
                    "response_split": "### 回答：",
                }
            else:
                raise ValueError(f"Unknown template: {template_name}")

    def generate_prompt(
        self,
        instruction: str,
        input: str = None,
        label: str = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        # 从模型的输出中提取最终的回答部分
        return output.split(self.template["response_split"])[1].strip()