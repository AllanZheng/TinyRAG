import pandas as pd
from pykeen.triples import TriplesFactory
from neo4j import GraphDatabase


# 示例三元组数据
triples = [
    ('Alice', 'knows', 'Bob'),
    ('Bob', 'likes', 'Carol'),
    ('Carol', 'works_with', 'Alice')
]

# 创建一个 DataFrame
df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])

# 使用 PyKEEN 创建三元组工厂
triples_factory = TriplesFactory.from_labeled_triples(df.values)

# Neo4j 连接信息
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))


def add_triples(tx, head, relation, tail):
    tx.run("MERGE (h:Entity {name: $head}) "
           "MERGE (t:Entity {name: $tail}) "
           "MERGE (h)-[r:Relation {name: $relation}]->(t)",
           head=head, relation=relation, tail=tail)


with driver.session() as session:
    for head, relation, tail in triples:
        session.write_transaction(add_triples, head, relation, tail)

driver.close()
    