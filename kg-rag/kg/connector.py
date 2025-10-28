from config.settings import settings
from py2neo import Graph


class Neo4jConnector:
    def __init__(self):
        self.graph = Graph(
            uri=settings.neo4j_uri,
            auth=settings.neo4j_auth
        )

    def query_subgraph(self, entities: list[str]) -> list:
        """²éÑ¯¶àÌø×ÓÍ¼"""
        cypher = f"""
        MATCH (e)-[r*1..2]->(t) 
        WHERE e.name IN {entities}
        RETURN e.name AS head, type(r[0]) AS rel, t.name AS tail
        LIMIT {settings.max_kg_triples}
        """
        return self.graph.run(cypher).data()
