import pandas as pd
import igraph as ig
from leidenalg import find_partition, ModularityVertexPartition

# 示例三元组数据 (10,000行)
triples = [
    ("实体A", "关系", "实体B"),
    ("实体B", "关系", "实体C"),
    # ... 其他三元组
]
df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

# 构建有向图（Leiden算法需无向图，需转换）
graph = ig.Graph.TupleList(
    df.itertuples(index=False), 
    directed=True,
    edge_attrs=["predicate"]
)
undirected_graph = graph.as_undirected()  # Leiden算法需无向图