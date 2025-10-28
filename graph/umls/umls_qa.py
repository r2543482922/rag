# -*- coding: utf-8 -*-
import re

import openai  # 可选
from flask import Flask, request, jsonify
from neo4j import GraphDatabase

# --- 配置 ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "lty20001114"
OPENAI_API_KEY = "sk-IPIz8j09SISTzvAu0hPgWH6e4Gc0Y6UICowjz4KOGIxmWC0w"  # 可选

openai.api_key = OPENAI_API_KEY

# --- 初始化 ---
app = Flask(__name__)


class Neo4jQA:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query_concept_by_name(self, name_zh):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)-[:HAS_LABEL]->(l:Label)
                WHERE l.name CONTAINS $name
                RETURN c.cui AS cui, c.name_zh AS name, c.source AS source
            """, name=name_zh)
            return [record.data() for record in result]

    def query_related_concepts(self, cui):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c1:Concept {cui: $cui})-[r:RELATED_TO]->(c2:Concept)
                RETURN c2.cui AS cui, c2.name_zh AS name, r.rel AS rel, r.rela AS rela
            """, cui=cui)
            return [record.data() for record in result]

    def query_concept_types(self, cui):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {cui: $cui})-[:HAS_TYPE]->(t:SemanticType)
                RETURN t.tui AS tui, t.name AS type
            """, cui=cui)
            return [record.data() for record in result]

    # 查询敏感性数据（扩展功能）
    def query_sensitivity_data(self, concept_name, antibiotic, time_point, strain, assay_type, method):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)-[:HAS_LABEL]->(l:Label)
                WHERE l.name = $concept_name
                MATCH (c)-[:RELATED_TO]->(s:Sensitivity)
                WHERE s.antibiotic = $antibiotic AND s.time_point = $time_point 
                AND s.strain = $strain AND s.assay_type = $assay_type AND s.method = $method
                RETURN s.min_inhibitory_concentration AS mic, s.result AS result
            """, concept_name=concept_name, antibiotic=antibiotic, time_point=time_point,
                                 strain=strain, assay_type=assay_type, method=method)
            return [record.data() for record in result]


neo4j_qa = Neo4jQA(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


# --- 路由 ---

@app.route('/')
def home():
    return "UMLS Q&A系统 - 欢迎访问！"


@app.route("/qa", methods=["POST"])
def answer_question():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 示例问题解析：氟氯西林
    match = re.match(r".*是(.+?)？?$", question)  # 匹配问题格式：什么是xxx
    if not match:
        return jsonify({"error": "无法理解问题格式"}), 400

    term = match.group(1).strip()

    # 查询概念
    concepts = neo4j_qa.query_concept_by_name(term)
    if not concepts:
        return jsonify({"answer": f"未找到“{term}”的医学概念。"}), 200

    # 提取相关的复合信息，例如“氟氯西林:抗生素敏感性:时间点:分离株:定量型或序数型:最小抑菌浓度法”
    concept_name = concepts[0]["name"]

    # 根据需要解析复合数据并返回简化的回答
    split_data = concept_name.split(":")  # 假设格式是“氟氯西林:抗生素敏感性:时间点:分离株:定量型或序数型:最小抑菌浓度法”
    simplified_concept_name = split_data[0]  # 获取氟氯西林部分

    # 获取相关类型和关系
    cui = concepts[0]["cui"]
    types = neo4j_qa.query_concept_types(cui)
    related = neo4j_qa.query_related_concepts(cui)

    answer = {
        "concept": {"name": simplified_concept_name},  # 返回简化的概念名称
        "types": types,
        "related": related
    }

    return jsonify({"answer": answer}), 200


# --- 入口 ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
