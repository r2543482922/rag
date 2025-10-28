import json

from py2neo import Graph

from question_classifier import QuestionClassifier
from question_parser import QuestionPaser


class JSONAnswerSearcher:
    def __init__(self):
        try:
            self.g = Graph(
                "bolt://localhost:7687",
                auth=("neo4j", "lty20001114"),
                secure=False
            )
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
            # "metadata": {
            #     "query_count": len(sqls),
            #     "result_count": 0
            # }
        }

        final_answers = []

        for sql_ in sqls:
            question_type = sql_['question_type']
            queries = sql_['sql']

            for query in queries:
                detail_item = {
                    "question_type": question_type,
                    # "cypher_query": query,
                    "nodes": [],
                    "relationships": [],
                    "raw_results": []
                }

                try:
                    ress = self.g.run(query).data()
                    # response["metadata"]["result_count"] += len(ress)

                    # 记录原始结果
                    detail_item["raw_results"] = ress

                    # 提取节点和关系信息
                    for record in ress:
                        node_info = {}
                        rel_info = {}

                        for key, value in record.items():
                            if key.startswith('m.') or key.startswith('n.'):
                                node_name = value
                                node_type = 'Disease' if 'm.name' in key else key.split('.')[0].upper()
                                node_details = self.get_node_details(node_name, node_type)

                                node_info = {
                                    # "identifier": key,
                                    "name": node_name,
                                    "label": node_type,
                                    "properties": node_details
                                }
                                detail_item["nodes"].append(node_info)

                            elif key == 'r.name':
                                rel_details = self.get_relationship_details(
                                    record.get('m.name', ''),
                                    record.get('n.name', ''),
                                    value
                                )
                                rel_info = {
                                    "type": value,
                                    "from": record.get('m.name', ''),
                                    "to": record.get('n.name', ''),
                                    "properties": rel_details
                                }
                                detail_item["relationships"].append(rel_info)

                    response["data"]["details"].append(detail_item)

                    # 生成友好回答
                    answers = self.answer_prettify(question_type, ress)
                    if answers:
                        final_answers.append(answers)

                except Exception as e:
                    error_item = {
                        "question_type": question_type,
                        # "cypher_query": query,
                        "error": str(e)
                    }
                    response["data"]["details"].append(error_item)
                    response["status"] = "partial_success"

        response["data"]["friendly_answer"] = '\n'.join(final_answers) if final_answers else '未找到相关信息'
        return response

    def answer_prettify(self, question_type, answers):
        # 保持原有的友好回答生成逻辑
        if not answers:
            return ''

        # 示例简化版，实际应包含完整逻辑
        if question_type == 'disease_symptom':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            return f'{subject}的症状包括：{"；".join(list(set(desc))[:self.num_limit])}'

        elif question_type == 'disease_desc':
            desc = [i['m.desc'] for i in answers]
            subject = answers[0]['m.name']
            return f'{subject},熟悉一下：{"；".join(list(set(desc))[:self.num_limit])}'

        # 其他问题类型的处理...
        return ''

    def get_node_details(self, node_name, label):
        """获取节点的完整详细信息"""
        query = f"MATCH (n:{label}) WHERE n.name = $name RETURN properties(n) as props"
        result = self.g.run(query, name=node_name).data()
        return result[0]['props'] if result else {}

    def get_relationship_details(self, start_node, end_node, rel_type):
        """获取关系的完整详细信息"""
        query = f"""
        MATCH (s)-[r:{rel_type}]->(e)
        WHERE s.name = $start_name AND e.name = $end_name
        RETURN properties(r) as rel_props
        """
        result = self.g.run(query, start_name=start_node, end_name=end_node).data()
        return result[0]['rel_props'] if result else {}


class JSONChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = JSONAnswerSearcher()
        self.default_answer = {
            "status": "success",
            "data": {
                "friendly_answer": "您好，我是医药智能助理，希望可以帮到您。如果我没有理解您的问题，请尝试换种方式提问。祝您身体棒棒！",
                "details": []
            },
            # "metadata": {
            #     "query_count": 0,
            #     "result_count": 0
            # }
        }

    def chat_main(self, sent):
        try:
            # 分类问题
            res_classify = self.classifier.classify(sent)
            if not res_classify:
                return self.default_answer

            # 生成查询
            res_sql = self.parser.parser_main(res_classify)
            if not res_sql:
                return self.default_answer

            # 执行查询并获取JSON格式结果
            return self.searcher.search_main(res_sql)

        except Exception as e:
            error_response = {
                "status": "error",
                "error_message": str(e),
                "data": {
                    "friendly_answer": "系统处理您的请求时出现问题",
                    "details": []
                },
                # "metadata": {
                #     "query_count": 0,
                #     "result_count": 0
                # }
            }
            return error_response


if __name__ == '__main__':
    handler = JSONChatBotGraph()

    print("医药知识图谱问答系统已启动(JSON模式)，输入'退出'或按Ctrl+C结束对话")
    print("=============================================")

    while True:
        try:
            question = input('用户: ').strip()
            if question.lower() in ['退出', 'exit', 'quit']:
                print(json.dumps({
                    "status": "success",
                    "message": "对话结束",
                    "data": None
                }, indent=2, ensure_ascii=False))
                break

            if not question:
                continue

            result = handler.chat_main(question)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("\n=============================================")

        except KeyboardInterrupt:
            print(json.dumps({
                "status": "success",
                "message": "对话结束",
                "data": None
            }, indent=2, ensure_ascii=False))
            break
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "error_message": str(e),
                "data": None
            }, indent=2, ensure_ascii=False))
