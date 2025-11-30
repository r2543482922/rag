# create_dpo_data.py (Updated for your Prompter structure)

import json
from typing import List, Dict, Optional
import os

# ------------------------------------------------------------------
# 导入 Prompter (假设它在 utils 文件夹中)
# ------------------------------------------------------------------
try:
    from utils.prompter import Prompter
except ImportError:
    # 如果无法导入，使用您提供的硬编码 Prompter 结构
    class Prompter:
        def __init__(self, template_name: str = "med_template", verbose: bool = False):
            # 简化结构，仅用于格式化
            self.template = {
                "prompt_input": "你是一位专业的医疗助手。请根据下面提供的指令和补充信息（包括知识库和历史对话）给出完整、准确的回答。\n\n### 指令：\n{instruction}\n\n### 补充信息：\n{input}\n\n### 回答：\n",
                "response_split": "### 回答：",
            }

        def generate_prompt(
                self, instruction: str, input: str = None, label: str = None
        ) -> str:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
            if label:
                res = f"{res}{label}"
            return res


# ------------------------------------------------------------------
# ⚠️ 替换区: 导入您实际的 RAG 检索器和 LLM 接口
# ------------------------------------------------------------------

class MockRetriever:  # 替换为您实际的 MedicalGraphRetriever
    def query_entity_context(self, entity_name):
        if entity_name == "卵巢小细胞癌":
            return "【疾病：卵巢小细胞癌】转移：转移部位手术效果可能不理想，一般不推荐。治疗：化疗、放疗。"
        if entity_name == "肝动脉瘤":
            return "【疾病：肝动脉瘤】并发症：梗阻性黄疸。诊治：需腹部平片、超声、CTA。治疗：可介入或手术治疗。"
        if entity_name == "反应性关节炎":
            return "【疾病：反应性关节炎】症状：关节红肿疼痛、排尿困难。治疗：沙利度胺、泼尼松。"
        if entity_name == "毛细胞白血病":
            return "【疾病：毛细胞白血病】多发群体：白人血统、男性、中年。"
        if entity_name == "转移性乳腺癌":
            return "【疾病：转移性乳腺癌】多发群体：女性。"
        return ""


class MockLLM:  # 替换为您实际的模型推理函数或API调用
    def generate(self, prompt: str, is_rag_enabled: bool, original_output: str) -> str:
        # 1. 生成 Preferred Answer (Chosen: RAG-Informed)
        if is_rag_enabled:
            if "卵巢小细胞癌" in prompt and "手术治疗" in prompt:
                return "根据知识库信息，当卵巢小细胞癌转移至其它部位时，手术治疗的效果可能不理想，因此一般不推荐进行手术治疗，可以采用化疗、放疗等综合治疗手段。"
            if "毛细胞白血病" in prompt and "转移性乳腺癌" in prompt and "多发群体" in prompt:
                return "知识库显示，毛细胞白血病的多发群体包括白人血统、男性、中年等；而转移性乳腺癌的多发群体则是女性。两者的多发群体存在明显差异。"

            # 使用原始 SFT 答案作为高质量 Chosen 答案
            return original_output

        # 2. 生成 Rejected Answer (Rejected: Generic/Ignoring RAG)
        else:
            if "肝动脉瘤" in prompt and "黄疸" in prompt:
                # 模拟通用但缺乏特异性的回答
                return "该患者病情复杂，建议尽快安排肝胆外科或介入科会诊。需要进行详细的影像学检查如CT或MRI，以便制定准确的治疗方案。请您遵循医嘱。"

            # 模拟模型给出模糊或不完整的回答
            return "您的提问涉及医学专业知识，建议您及时到专科医院进行全面检查，避免耽误病情。"


# ------------------------------------------------------------------

# 实例化组件
retriever = MockRetriever()
llm = MockLLM()
prompter = Prompter(template_name="med_template")  # 使用您的 Prompter


def get_entities(text: str) -> List[str]:
    """ ⚠️ 实际应用中，请替换为您的 AliyunNERExtractor 逻辑 """
    entities = []
    if "卵巢小细胞癌" in text: entities.append("卵巢小细胞癌")
    if "肝动脉瘤" in text: entities.append("肝动脉瘤")
    if "反应性关节炎" in text: entities.append("反应性关节炎")
    if "毛细胞白血病" in text: entities.append("毛细胞白血病")
    if "转移性乳腺癌" in text: entities.append("转移性乳腺癌")
    return entities


def generate_dpo_data_point_single_turn(
        data_point: Dict,
        system_instruction: str
) -> Optional[Dict]:
    current_question = data_point["instruction"]
    original_output = data_point["output"]

    # A. NER 提取实体
    entities = get_entities(current_question)

    # B. RAG 检索
    kg_context = ""
    if entities:
        info = retriever.query_entity_context(entities[0])
        if info: kg_context = info

    # 1. 构造 Prompt 的两个核心部分
    # INSTRUCTION: 包含核心的系统指令和用户问题
    instruction_part = f"{system_instruction}\n\n用户问题: {current_question}"

    # INPUT: 包含 RAG Context，用于 Preferred Response 的训练
    input_part = f"【知识库信息】:\n{kg_context or '暂无有效知识库信息'}"

    # 2. 格式化最终 DPO Prompt (使用 Prompter)
    final_prompt = prompter.generate_prompt(
        instruction=instruction_part,
        input=input_part,
        label=None  # DPO prompt 中不含 label
    )

    # 3. LLM 生成 Chosen 和 Rejected

    # Chosen (Preferred) Response: RAG 增强后的回答 (is_rag_enabled=True)
    chosen_response = llm.generate(final_prompt, is_rag_enabled=True, original_output=original_output)

    # Rejected Response: 模拟 RAG 失败或模型失常的通用回答 (is_rag_enabled=False)
    rejected_response = llm.generate(final_prompt, is_rag_enabled=False, original_output=original_output)

    # 过滤：如果 Chosen 和 Rejected 答案相同，这条数据无效
    if chosen_response.strip() == rejected_response.strip():
        return None

    # 最终输出格式：符合 DPOTrainer 要求
    return {
        "prompt": final_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response
    }


def create_single_turn_dpo_dataset(sft_data: List[Dict], output_file: str):
    DPO_DATASET = []

    # DPO 训练中，system_instruction 必须明确要求使用知识库
    system_instruction = (
        "你的主要目标是根据【知识库信息】回答用户问题。\n"
        "规则：1. 必须优先引用知识库信息。2. 给出具体的医疗建议，避免通用建议。"
    )

    # 遍历 SFT 数据集，转化为 DPO 格式
    for data_point in sft_data:
        dpo_item = generate_dpo_data_point_single_turn(data_point, system_instruction)
        if dpo_item:
            DPO_DATASET.append(dpo_item)

    # 保存为 JSONL 格式
    with open(output_file, "w", encoding="utf-8") as f:
        for item in DPO_DATASET:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 单轮 DPO 偏好数据集已生成，共 {len(DPO_DATASET)} 条，文件名为 {output_file}。")


if __name__ == "__main__":
    # 您的原始 SFT 数据 (作为输入)
    sft_input_data = [
        {"instruction": "一名年龄在70岁的女性，出现了晕厥、不自主颤抖、情绪不稳等症状，请详细说明其手术治疗和术前准备。",
         "input": "",
         "output": "该病需要进行电极导线、脉冲发生器和永久心脏起搏器置入术，并需要使用镇静药物和局麻对病人进行手术治疗。术前准备包括1-3天的时间进行术前检查和生活方式的调整。"},
        {"instruction": "一个患者的卵巢小细胞癌转移至其它部位，是否有必要进行手术治疗？", "input": "",
         "output": "当卵巢小细胞癌转移至其它部位时，手术治疗的效果可能不理想，因此一般不推荐进行手术治疗。针对转移病灶，可以采用化疗、放疗等治疗手段进行综合治疗。"},
        {"instruction": "一位60岁男性患者由于肝动脉瘤出现肝功能异常，具体表现为黄疸，该如何诊治？", "input": "",
         "output": "该患者可能出现了肝动脉瘤并发症之一的梗阻性黄疸，需要进行腹部平片、彩色多普勒超声、CTA、动脉造影等检查后确定诊断，治疗方面应按照病因治疗，如若为肝动脉炎、动脉粥样硬化等病因，应该采取手术治疗，而若是单纯的肝动脉瘤，可以进行介入治疗或手术治疗。"},
        {"instruction": "患者男，28岁，关节部位红肿疼痛，排尿困难，近期有过微生物感染史。请问可能患的是什么疾病？如何治疗？",
         "input": "", "output": "可能患有反应性关节炎。治疗方案可以采用沙利度胺、泼尼松等药物。同时还要注意休息和饮食。"},
        {
            "instruction": "一名患者因腹胀、纳差等症状就诊，检查发现肝大、腹水及肝功能损害，诊断为肝淀粉样变性，请问该患者应该接受哪些治疗方案？",
            "input": "", "output": "该患者应该接受弥漫性系膜硬化、苯丙氨酸氮芥等治疗方案。"},
        {"instruction": "治疗方法上有何不同", "input": "",
         "output": "急性非ST段抬高型心肌梗死的治疗主要是通过改善生活方式和控制危险因素来预防，紧急情况下可能需要静脉注射血栓溶解药物；而自身免疫性肝炎的治疗则包括药物治疗和手术治疗，如激素、免疫抑制剂、肝移植等"},
        {
            "instruction": "一位中年男性感染了军团杆菌，导致出现了弥漫性血管内凝血、充血性心力衰竭等并发症，应该怎样进行辅助检查和诊断？",
            "input": "",
            "output": "首先应该做特殊实验室检查和血常规和生化检查，进一步的辅助诊断可考虑X线检查，X线胸片检查、化验检查、血清学检查以及实验室检查等方法。"},
        {"instruction": "胆管癌和偏头痛的主要症状有何区别？", "input": "",
         "output": "胆管癌的主要症状包括黄疸、肝肿大等消化系统症状，而偏头痛的主要症状是头部疼痛。"},
        {"instruction": "胡女士出现了黄疸、恶心、呕吐等症状，可能患了什么疾病？", "input": "",
         "output": "胡女士可能患有胆道张力低下综合征。建议尽快到内科进行检查，并进行西医和中医治疗。"},
        {
            "instruction": "一位妊娠31周孕妇在家中自行服用草药，随后出现呕吐、腹痛、恶心、疲劳、皮肤瘙痒、黄疸等症状，血液检查显示血清转氨酶升高，交叉免疫电泳检测显示未见明显异常。 ",
            "input": "",
            "output": "可能是由于在未咨询医生的情况下自行服用草药导致的急性妊娠期肝内胆汁淤积症。建议及时就医，如需紧急治疗可在医生指导下使用胆酸螯合剂、皮质激素等药物，并密切监测胎儿状态与身体其他部位不适情况。"},
        {"instruction": "毛细胞白血病和转移性乳腺癌的多发群体有什么不同？", "input": "",
         "output": "毛细胞白血病的多发群体包括白人血统、男性、中年等，而转移性乳腺癌的多发群体则是女性。 "},
    ]

    create_single_turn_dpo_dataset(sft_input_data, "single_turn_dpo_preference_data.jsonl")