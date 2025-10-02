from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Embeddings import JinaEmbedding,RemoteBgeEmbedding,BgeEmbedding,ZhipuEmbedding
from RAG.Reranker import BgeReranker
import  time
# 没有保存数据库
# docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
# vector = VectorStore(docs)
# embedding = BgeEmbedding() # 创建EmbeddingModel
# vector.get_vector(EmbeddingModel=embedding)
# vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# # vector.load_vector('./storage') # 加载本地的数据库

# question = '正向扫描的原理是什么？'

# content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
# chat = OpenAIChat(model='gpt-3.5-turbo-1106')
# print(chat.chat(question, [], content))




start_load_rank_time =  time.time()
reranker = BgeReranker()
end_time = time.time()
print("加载reranker耗费时间:",end_time-start_load_rank_time)
# 保存数据库之后
start_load_vector_time =  time.time()
vector = VectorStore()

vector.load_vector('./storage') # 加载本地的数据库
end_time = time.time()
print("加载vector耗费时间:",end_time-start_load_vector_time)
question = '用中文介绍下git'
start_query_time =  time.time()
# embedding = ZhipuEmbedding() # 创建EmbeddingModel
bgeEmbedding = BgeEmbedding(path="C:\\Users\\Allan\\.cache\\huggingface\\hub\\models--base--finetune-english")
content = vector.query(question, EmbeddingModel=bgeEmbedding, k=3)
content = vector.convarinace_query(question, EmbeddingModel=bgeEmbedding, k=3)
end_time = time.time()
print("相似度耗费时间:",end_time-start_query_time)
start_rank_time =  time.time()
rerank_content = reranker.rerank(question, content, k=2)
# 最后选择最相似的文档, 交给LLM作为可参考上下文
best_content = rerank_content[0]
# chat = OpenAIChat(model='deepseek-r1-32b')
end_time = time.time()
print("重排耗费时间:",end_time-start_rank_time)
chat = OpenAIChat(model='qwen3-32b')

print(question)
# print(content)
print(rerank_content)
print(chat.chat(question, [], rerank_content))

