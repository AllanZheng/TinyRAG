# 安装依赖：pip install networkx matplotlib
import networkx as nx
from collections import defaultdict
# 安装依赖：pip install pykeen py2neo
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from py2neo import Graph
import numpy as np
from neo4j import GraphDatabase
import torch
import  re
# data = np.array([('a',1,'b'),('a',2,'c'),('b',2,'c')])

def fast_triple_to_graph(triples):
    """基于字典和NetworkX的快速转换（时间复杂度O(n)）"""
    graph = nx.MultiDiGraph()
    entity_counter = defaultdict(int)
    
    for head, relation, tail in triples:
        # 添加节点和边（自动去重）
        graph.add_node(head, type="Entity", count=entity_counter[head])
        graph.add_node(tail, type="Entity", count=entity_counter[tail])
        graph.add_edge(head, tail, label=relation, key=f"{head}_{relation}_{tail}")
        entity_counter[head] += 1
        entity_counter[tail] += 1
    return graph
def data_clean(stri):
    s= re.sub(r' +', ' ', stri)
    s=s.replace("’", "_").replace("'", "_").replace(" ", "_")
    first_index = s.find('(')
    last_index = s.rfind(')')
    if first_index != -1 and last_index != -1 and first_index < last_index:
        return s[:first_index] + s[first_index + 1:last_index] + s[last_index + 1:]
    return s


def file_read_triple(path):
    triples = []
    with open(path, 'r',encoding="utf-8") as file:
        for line in file:
        # 去除首尾空格并分割字段（兼容空格、制表符、多空格）
            if line!="" or line!="/n":
                data =data_clean(line)
                fields = data.split(",")
                if len(fields) == 3:
                    # for i in fields:
                    #     i = i
                    triples.append(fields)
                    # print(fields)  # 输出示例：[['Tom', 'likes', 'apples'], ...]
        return triples


data = file_read_triple("D:\\Git\\TinyRAG\\triple_data\\kgs1.txt")
print("success")
triples=np.array(data)
# 生成知识图谱嵌入向量
# triples = np.array([
#     ("苹果", "生产", "iPhone12"),
#     ("谷歌", "开发", "Android"),
#     ("微软", "收购", "GitHub"),
#     ("Paris", "is", "capital of France"),
#     ("Eiffel Tower", "located in", "Paris"),
#     ("Ship accidents in East China Sea", "caused by", "challenging navigation") 
# ])
# triples=np.array([('a',1,'b'),('a',2,'c'),('b',2,'c'),('c',2,'d'),('d',2,'b')])
# 自动分割训练集和测试集（比例可调）
tf = TriplesFactory.from_labeled_triples(
    triples=triples,
    create_inverse_triples=False  # 是否创建反向关系

)
# training, testing = tf.split([.8, .2]) 
# 检查 GPU 是否可用
device1 = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device1}")
# 训练TransE嵌入模型
result = pipeline(
    training=tf,
    testing=tf,
    model='TransE',
    epochs=5,
    dimensions=100,  # 高维向量提升语义表达[4,8](@ref)
    device=device1
)

# 保存嵌入向量
result.save_to_directory('output/knowledge_test')

# 连接Neo4j图数据库（需提前启动neo4j服务）
neo4j_graph = Graph("bolt://localhost:7687", auth=("neo4j", "2025"))

# 将嵌入向量存入图数据库
node,rel_count,err_count=0,0,0
# for entity_id, entity_name in result.training.entity_id_to_label.items():
#     cpu_ten=result.model.entity_representations[0].cpu()
#     embedding = cpu_ten._embeddings.weight[entity_id].detach().numpy()
#     # entity_name=entity_name.replace("'", "\\'")
#     cypher = f"""
#     MERGE (e:Entity {{name: '{entity_name}'}})
#     SET e.embedding = {embedding.tolist()}
#     """
#     try:
#         neo4j_graph.run(cypher)
#     except Exception as e:
#         print("create node error:",str(e))
#         err_count+1
        
#     node=node+1
#     if node%1000==0:
#         print(node)

# 添加关系（带概率权重）
for (head, relation, tail) in triples:
    head_id = result.training.entity_to_id[head]
    relation_id = result.training.relation_to_id[relation]
    tail_id = result.training.entity_to_id[tail]
    hrt_batch = np.array([[head_id, relation_id, tail_id]])
    # try:
    hrt_batch_tensor = torch.from_numpy(hrt_batch).to("cuda")
    scores = result.model.predict(hrt_batch= hrt_batch_tensor, target='relation')
    prob = torch.max(scores).item()
    # relation=relation.replace("'", "\\'")
    processed_relation = f"REL_{relation}"
    try:
        neo4j_graph.run(f"""
        MATCH (h:Entity {{name: '{head}'}}), (t:Entity {{name: '{tail}'}})
        MERGE (h)-[r:{processed_relation} {{probability: {prob}}}]->(t)
        """)
    except Exception as e:
        print("create relation error:",str(e))
        err_count+1
    rel_count=rel_count+1
    if rel_count%1000==0:
        print(rel_count)


print("nodes:",node,"relations_count:",rel_count,"err_count:",err_count)

    # except Exception as e:
    #     print(f"Error processing triple ({head}, {relation}, {tail}): {e}")
    #     break

# 示例三元组数据
# triples = [
#     ("苹果", "生产", "iPhone12"),
#     ("谷歌", "开发", "Android"),
#     ("微软", "收购", "GitHub")
# ]

# # # 执行转换
# kg_graph = fast_triple_to_graph(triples)

# # 可视化图谱（可选）
# nx.draw(kg_graph, with_labels=True, node_color='skyblue')
# plt.show()

# # 导出为图文件（GEXF格式）
# nx.write_gexf(kg_graph, "fast_graph.gexf")