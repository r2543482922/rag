from ragatouille import RAGPretrainedModel
from .connector import Neo4jConnector
from transformers import AutoTokenizer
import torch


class KGRetriever:
    def __init__(self):
        self.rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        self.kg = Neo4jConnector()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    def retrieve(self, question: str) -> dict:
        # 文本检索
        text_results = [r["content"] for r in self.rag.search(question, k=2)]

        # 实体识别（简化版，实际可用NER模型）
        entities = list(set([
            word for word in question.split()
            if len(word) > 1 and not self.tokenizer.is_special_token(word)
        ]))

        # 知识图谱查询
        kg_triples = self.kg.query_subgraph(entities)

        return {
            "texts": text_results,
            "kg_triples": kg_triples
        }