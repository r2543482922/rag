import sys
from typing import List, Dict

import fire
import numpy as np
import torch
import wandb
from datasets import load_dataset, Dataset
from loguru import logger
from neo4j import GraphDatabase
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)

from utils.prompter import Prompter


class MedicalKnowledgeGraph:
    def __init__(self):
        self.uri = "neo4j://localhost:7687"
        self.user = "neo4j"
        self.password = "lty20001114"
        self.driver = None
        self._setup_logging()
        self.connect()

    def _setup_logging(self):
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=30
            )
            logger.success("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {str(e)}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("已关闭Neo4j连接")

    def fetch_comprehensive_knowledge(self, disease_name: str) -> Dict:
        """从Neo4j获取完整的医疗知识图谱数据"""
        if not self.driver:
            self.connect()

        knowledge = {
            # 节点实体
            "drugs": [],  # 药品
            "foods": [],  # 食物
            "checks": [],  # 检查
            "departments": [],  # 科室
            "producers": [],  # 药品大类
            "symptoms": [],  # 症状
            "diseases": [],  # 相关疾病
            "disease_info": {},  # 疾病详情

            # 实体关系
            "rels_recommandeat": [],  # 推荐食谱
            "rels_noteat": [],  # 忌吃
            "rels_doeat": [],  # 宜吃
            "rels_department": [],  # 科室关系
            "rels_commonddrug": [],  # 常用药品
            "rels_drug_producer": [],  # 药品厂商
            "rels_recommanddrug": [],  # 好评药品
            "rels_check": [],  # 诊断检查
            "rels_symptom": [],  # 症状
            "rels_acompany": [],  # 并发症
            "rels_category": []  # 所属科室
        }

        try:
            with self.driver.session() as session:
                # 获取疾病基本信息
                disease_query = """
                MATCH (d:Disease {name: $disease_name})
                RETURN d
                """
                disease_data = session.run(disease_query, {"disease_name": disease_name}).data()
                if disease_data:
                    knowledge["disease_info"] = dict(disease_data[0]['d'])

                # 获取相关药品
                drug_queries = {
                    "rels_commonddrug": "MATCH (d:Disease {name: $disease_name})-[:common_drug]->(drug:Drug) RETURN drug.name as name",
                    "rels_recommanddrug": "MATCH (d:Disease {name: $disease_name})-[:recommand_drug]->(drug:Drug) RETURN drug.name as name",
                    "rels_drug_producer": """
                    MATCH (d:Disease {name: $disease_name})-[:common_drug|recommand_drug]->(drug:Drug)-[:drugs_of]->(prod:Producer)
                    RETURN prod.name as producer, drug.name as drug
                    """
                }

                for rel_type, query in drug_queries.items():
                    result = session.run(query, {"disease_name": disease_name}).data()
                    knowledge[rel_type] = result
                    if rel_type != "rels_drug_producer":
                        knowledge["drugs"].extend([item['name'] for item in result])

                # 获取食物关系
                food_queries = {
                    "rels_recommandeat": "MATCH (d:Disease {name: $disease_name})-[:recommand_eat]->(food:Food) RETURN food.name as name",
                    "rels_noteat": "MATCH (d:Disease {name: $disease_name})-[:no_eat]->(food:Food) RETURN food.name as name",
                    "rels_doeat": "MATCH (d:Disease {name: $disease_name})-[:do_eat]->(food:Food) RETURN food.name as name"
                }

                for rel_type, query in food_queries.items():
                    result = session.run(query, {"disease_name": disease_name}).data()
                    knowledge[rel_type] = result
                    knowledge["foods"].extend([item['name'] for item in result])

                # 获取检查项目
                check_query = """
                MATCH (d:Disease {name: $disease_name})-[:need_check]->(check:Check)
                RETURN check.name as name
                """
                knowledge["rels_check"] = session.run(check_query, {"disease_name": disease_name}).data()
                knowledge["checks"] = [item['name'] for item in knowledge["rels_check"]]

                # 获取症状
                symptom_query = """
                MATCH (d:Disease {name: $disease_name})-[:has_symptom]->(symptom:Symptom)
                RETURN symptom.name as name
                """
                knowledge["rels_symptom"] = session.run(symptom_query, {"disease_name": disease_name}).data()
                knowledge["symptoms"] = [item['name'] for item in knowledge["rels_symptom"]]

                # 获取并发症
                acompany_query = """
                MATCH (d:Disease {name: $disease_name})-[:acompany_with]->(other:Disease)
                RETURN other.name as name
                """
                knowledge["rels_acompany"] = session.run(acompany_query, {"disease_name": disease_name}).data()
                knowledge["diseases"] = [item['name'] for item in knowledge["rels_acompany"]]

                # 获取科室信息
                department_query = """
                MATCH (d:Disease {name: $disease_name})-[:belongs_to]->(dept:Department)
                OPTIONAL MATCH (dept)-[:belongs_to]->(parent:Department)
                RETURN dept.name as department, parent.name as parent_department
                """
                dept_result = session.run(department_query, {"disease_name": disease_name}).data()
                knowledge["rels_category"] = []
                knowledge["rels_department"] = []

                for item in dept_result:
                    if item['department']:
                        knowledge["departments"].append(item['department'])
                        knowledge["rels_category"].append({
                            "disease": disease_name,
                            "department": item['department']
                        })

                        if item['parent_department']:
                            knowledge["departments"].append(item['parent_department'])
                            knowledge["rels_department"].append({
                                "child": item['department'],
                                "parent": item['parent_department']
                            })

            logger.success(f"成功获取疾病'{disease_name}'的完整知识图谱数据")
            return knowledge

        except Exception as e:
            logger.error(f"获取知识图谱数据失败: {str(e)}")
            return knowledge

    def generate_knowledge_context(self, knowledge: Dict) -> str:
        """生成综合知识上下文文本"""
        context = "医疗知识上下文:\n\n"
        disease_info = knowledge.get("disease_info", {})

        # 疾病基本信息
        if disease_info:
            context += f"疾病名称: {disease_info.get('name', '')}\n"
            context += f"描述: {disease_info.get('desc', '')}\n"
            context += f"预防措施: {disease_info.get('prevent', '')}\n"
            context += f"病因: {disease_info.get('cause', '')}\n"
            context += f"易感人群: {disease_info.get('easy_get', '')}\n"
            context += f"治疗周期: {disease_info.get('cure_lasttime', '')}\n"
            context += f"治愈概率: {disease_info.get('cured_prob', '')}\n"
            context += f"治疗方式: {disease_info.get('cure_way', '')}\n\n"

        # 症状信息
        if knowledge.get("symptoms"):
            context += "症状:\n"
            context += "\n".join(f"- {symptom}" for symptom in knowledge["symptoms"]) + "\n\n"

        # 科室信息
        if knowledge.get("departments"):
            context += "相关科室:\n"
            dept_set = set(knowledge["departments"])
            context += "\n".join(f"- {dept}" for dept in dept_set) + "\n\n"

        # 药品信息
        if knowledge.get("drugs"):
            context += "相关药品:\n"
            drug_set = set(knowledge["drugs"])
            context += "\n".join(f"- {drug}" for drug in drug_set) + "\n\n"

            # 药品厂商关系
            if knowledge.get("rels_drug_producer"):
                context += "药品生产信息:\n"
                for item in knowledge["rels_drug_producer"]:
                    context += f"- {item.get('drug', '')} 由 {item.get('producer', '')} 生产\n"
                context += "\n"

        # 食物信息
        if knowledge.get("foods"):
            food_set = set(knowledge["foods"])

            # 忌吃食物
            if knowledge.get("rels_noteat"):
                context += "忌吃食物:\n"
                noteat = set(item['name'] for item in knowledge["rels_noteat"])
                context += "\n".join(f"- {food}" for food in noteat) + "\n\n"

            # 宜吃食物
            if knowledge.get("rels_doeat"):
                context += "宜吃食物:\n"
                doeat = set(item['name'] for item in knowledge["rels_doeat"])
                context += "\n".join(f"- {food}" for food in doeat) + "\n\n"

            # 推荐食物
            if knowledge.get("rels_recommandeat"):
                context += "推荐食物:\n"
                recommandeat = set(item['name'] for item in knowledge["rels_recommandeat"])
                context += "\n".join(f"- {food}" for food in recommandeat) + "\n\n"

        # 检查项目
        if knowledge.get("checks"):
            context += "诊断检查:\n"
            context += "\n".join(f"- {check}" for check in set(knowledge["checks"])) + "\n\n"

        # 相关疾病
        if knowledge.get("diseases"):
            context += "相关疾病:\n"
            context += "\n".join(f"- {disease}" for disease in set(knowledge["diseases"])) + "\n"

        return context


class KnowledgeGraphEnhancedTrainer:
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self._setup_logging()

    def _setup_logging(self):
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )

    def _load_and_preprocess_data(self, data_path: str) -> Dataset:
        """加载和预处理数据集"""
        try:
            if data_path.endswith((".json", ".jsonl")):
                dataset = load_dataset("json", data_files=data_path)
            else:
                dataset = load_dataset(data_path)
            return dataset
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    def _initialize_model(self, base_model: str, device_map: str = "auto") -> tuple:
        """初始化模型和tokenizer，使用4-bit量化"""
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(base_model)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            return model, tokenizer
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

    def _compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # 只计算非-100的标签
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted")
        }

    def train(
            self,
            base_model: str = "meta-llama/Llama-2-7b-hf",
            data_path: str = "data-literature/liver_cancer.json",
            output_dir: str = "./med-lora",
            batch_size: int = 128,
            micro_batch_size: int = 4,
            num_epochs: int = 10,
            learning_rate: float = 2e-5,
            cutoff_len: int = 512,
            val_set_size: int = 500,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],
            train_on_inputs: bool = False,
            group_by_length: bool = True,
            wandb_project: str = "llama_med",
            wandb_run_name: str = "",
            resume_from_checkpoint: str = None,
            prompt_template_name: str = "med_template",
            early_stopping_patience: int = 3,
            knowledge_integration: bool = True,
            disease_name: str = "肝癌"
    ):
        """增强的训练函数，整合知识图谱"""

        # 初始化
        gradient_accumulation_steps = batch_size // micro_batch_size
        prompter = Prompter(prompt_template_name)

        # 知识图谱集成
        knowledge_context = ""
        if knowledge_integration:
            logger.info(f"从Neo4j获取疾病'{disease_name}'的知识图谱数据...")
            knowledge = self.kg.fetch_comprehensive_knowledge(disease_name)
            knowledge_context = self.kg.generate_knowledge_context(knowledge)
            logger.info("知识图谱上下文:\n" + knowledge_context[:500] + "...")  # 打印前500字符

        # 加载模型
        logger.info("初始化模型...")
        model, tokenizer = self._initialize_model(base_model)

        # 准备模型
        model = prepare_model_for_int8_training(model)

        # 配置LoRA
        logger.info("配置LoRA...")
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.config.use_cache = False

        # 数据处理函数
        def tokenize(prompt, add_eos_token=True):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                    result["input_ids"][-1] != tokenizer.eos_token_id
                    and len(result["input_ids"]) < cutoff_len
                    and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()
            return result

        def generate_and_tokenize_prompt(data_point):
            """增强的提示生成，整合知识图谱上下文"""
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"]
            )

            # 整合知识图谱上下文
            if knowledge_context:
                full_prompt = knowledge_context + "\n\n" + full_prompt

            tokenized_full_prompt = tokenize(full_prompt)

            if not train_on_inputs:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"],
                    data_point["input"]
                )
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_prompt["labels"] = (
                        [-100] * user_prompt_len
                        + tokenized_full_prompt["labels"][user_prompt_len:]
                )

            return tokenized_full_prompt

        # 加载数据集
        logger.info("加载数据集...")
        dataset = self._load_and_preprocess_data(data_path)

        # 数据集分割
        if val_set_size > 0:
            train_val = dataset["train"].train_test_split(
                test_size=val_set_size,
                shuffle=True,
                seed=42
            )
            train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = None

        # 训练参数
        training_args = TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            report_to="wandb" if wandb_project else None,
            run_name=wandb_run_name,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            disable_tqdm=False,
        )

        # 回调函数
        callbacks = []
        if val_set_size > 0 and early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience
                )
            )

        # 训练器
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            callbacks=callbacks,
            compute_metrics=self._compute_metrics if val_set_size > 0 else None,
        )

        # 训练
        logger.info("开始训练...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # 保存最终模型
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.success(f"训练完成，模型已保存到 {output_dir}")
        self.kg.close()


if __name__ == "__main__":
    trainer = KnowledgeGraphEnhancedTrainer()
    fire.Fire(trainer.train)
