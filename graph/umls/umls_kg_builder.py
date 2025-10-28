# -*- coding: utf-8 -*-

import pandas as pd
from neo4j import GraphDatabase

# --- 配置区 ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "lty20001114"


# --- 加载 MRCONSO.RRF，仅保留中文 ---
def load_mrconso(path):
    cols = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI",
        "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"
    ]
    df = pd.read_csv(path, sep='|', names=cols, dtype=str, index_col=False)
    df = df[df["LAT"] == "CHI"]  # 精准匹配，避免 startswith
    df["LAT"] = "CHS"
    df = df[["CUI", "STR", "SAB", "LAT"]].drop_duplicates()
    return df


# --- 合并为 Concept 主节点 ---
def extract_concepts(df):
    df = df[df["LAT"] == "CHS"]
    return df.drop_duplicates(subset=["CUI", "SAB"]).rename(columns={"STR": "name_zh"})


# --- 加载 MRREL.RRF ---
def load_mrrel(path, valid_cuis):
    cols = [
        "CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA",
        "RUI", "SRUI", "SAB", "SL", "RG", "DIR", "SUPPRESS", "CVF"
    ]
    df = pd.read_csv(path, sep='|', names=cols, dtype=str, index_col=False)
    df = df[["CUI1", "CUI2", "REL", "RELA"]].drop_duplicates()
    return df[df["CUI1"].isin(valid_cuis) & df["CUI2"].isin(valid_cuis)]


# --- 加载 MRSTY.RRF ---
def load_mrsty(path, valid_cuis):
    cols = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
    df = pd.read_csv(path, sep='|', names=cols, dtype=str, index_col=False)
    df = df[["CUI", "TUI", "STY"]].drop_duplicates()
    return df[df["CUI"].isin(valid_cuis)]


# --- Neo4j 图谱构建类 ---
class UMLSGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_concepts(self, concepts_df, batch_size=1000):
        with self.driver.session() as session:
            for i in range(0, len(concepts_df), batch_size):
                batch = concepts_df.iloc[i:i + batch_size]
                with session.begin_transaction() as tx:
                    print(f"插入 Concept 节点 {i}-{i + batch_size}")
                    tx.run("""
                        UNWIND $rows AS row
                        MERGE (c:Concept {cui: row.CUI})
                        SET c.name_zh = row.name_zh, c.source = row.SAB
                    """, rows=batch.to_dict("records"))

    def create_labels(self, labels_df, batch_size=1000):
        labels_df = labels_df[labels_df["LAT"] == "CHS"]
        with self.driver.session() as session:
            for i in range(0, len(labels_df), batch_size):
                batch = labels_df.iloc[i:i + batch_size]
                with session.begin_transaction() as tx:
                    print(f"插入 Label 节点 {i}-{i + batch_size}")
                    tx.run("""
                        UNWIND $rows AS row
                        MATCH (c:Concept {cui: row.CUI})
                        MERGE (l:Label {name: row.STR, lang: 'CHS'})
                        MERGE (c)-[:HAS_LABEL]->(l)
                    """, rows=batch.to_dict("records"))

    def create_types(self, types_df, batch_size=1000):
        with self.driver.session() as session:
            for i in range(0, len(types_df), batch_size):
                batch = types_df.iloc[i:i + batch_size]
                print(f"插入语义类型 {i}-{i + batch_size}")
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $rows AS row
                        MERGE (s:SemanticType {tui: row.TUI})
                        ON CREATE SET s.name = row.STY
                        WITH s, row
                        MATCH (c:Concept {cui: row.CUI})
                        MERGE (c)-[:HAS_TYPE]->(s)
                    """, rows=batch.to_dict("records"))

    def create_relationships(self, rel_df, batch_size=1000):
        with self.driver.session() as session:
            # 过滤掉 'rela' 为 NaN 的行
            rel_df = rel_df.dropna(subset=['RELA'])
            # 或者，使用默认值替换 'rela' 为 NaN 的行
            # rel_df['RELA'] = rel_df['RELA'].fillna('UNKNOWN')

            for i in range(0, len(rel_df), batch_size):
                batch = rel_df.iloc[i:i + batch_size]
                print(f"插入关系 {i}-{i + batch_size}")
                with session.begin_transaction() as tx:
                    tx.run("""
                        UNWIND $rows AS row
                        MATCH (c1:Concept {cui: row.CUI1})
                        MATCH (c2:Concept {cui: row.CUI2})
                        MERGE (c1)-[:RELATED_TO {rel: row.REL, rela: row.RELA}]->(c2)
                    """, rows=batch.to_dict("records"))

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")


# --- 主流程 ---
if __name__ == "__main__":
    print("加载数据中...")
    conso_df = load_mrconso("../data/MRCONSO.RRF")
    concepts_df = extract_concepts(conso_df)
    valid_cuis = set(concepts_df["CUI"])

    rel_df = load_mrrel("../data/MRREL.RRF", valid_cuis)
    sty_df = load_mrsty("../data/MRSTY.RRF", valid_cuis)

    print(f"概念数: {len(concepts_df)}")
    print(f"标签数（中文）: {len(conso_df)}")
    print(f"关系数: {len(rel_df)}")
    print(f"语义类型数: {len(sty_df)}")

    print("导入 Neo4j 图谱中...")
    builder = UMLSGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # builder.clear_database()
    # builder.create_concepts(concepts_df)
    # builder.create_labels(conso_df)
    # builder.create_types(sty_df)
    builder.create_relationships(rel_df)
    builder.close()
    print("图谱导入完成！")
