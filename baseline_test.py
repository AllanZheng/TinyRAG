import os
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Embeddings import BgeEmbedding
from RAG.Reranker import BgeReranker
from evaluate.ragas_test import Extra_evaluate
import json
import time
import re
from datasets import Dataset
# 未创建向量数据库、更换数据集、或更换其他Embeddings模型后将have_created_db设置为False
# 运行一次成功创建向量数据库后可以将have_created_db设置为True
have_created_db = True
vector_path='vector/base' #/vector/kg_ocean/en ocean

# 创建EmbeddingModel
embedding = BgeEmbedding()
# 创建RerankerModel
reranker = BgeReranker()
# # 创建RerankerModel
# reranker = BgeReranker()
# 去除分离深度思考的内容
def remove_start_end(s):
    # start = "<think>"
    # end = "</think>"
    # s=s.replace('\r\n', '').replace('\n', '')
    return re.sub(r'.*?</think>', '', s,flags=re.DOTALL)

if have_created_db:
    # 保存数据库之后
    vector = VectorStore()
    vector.load_vector(vector_path)  # 加载本地的数据库
else:
    # 没有保存数据库
    start_time = time.time()
    docs = ReadFiles('train_data/ocean').get_content(max_token_len=600, cover_content=150)  # 获得data目录下的所有文件内容并分割
    time_1=time.time()
    print("读取文件花费的时间",time_1-start_time)
    vector = VectorStore(docs)

    vector.get_vector(EmbeddingModel=embedding)
    time_2=time.time()
    print("转换成向量花费的时间",time_2-time_1)
    vector.persist(path='vector/base')  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库
    time_3=time.time()
    print("向量存储到本地花费的时间",time_3-time_2)


question = 'Please introduce the knowledge about ship fire safety in the East China Sea area.'


# embedding = ZhipuEmbedding() # 创建EmbeddingModel
bgeEmbedding = BgeEmbedding()
# content = vector.query(question, EmbeddingModel=bgeEmbedding, k=1)
chat = OpenAIChat(model='Qwen3-32B')
# print(question)
# print(content)
# ans=remove_start_end(chat.chat(question, [], content))
# print(ans)

dicts={
"question": [],
"answer": [],
"contexts": [],
"reference":[]
}
# try:
print("开始")
with open("result/test_base_10.json","w",encoding='utf-8') as result_file,open("test_data/ocean/test10_base_random.json", 'r',encoding="utf-8") as que_file:
    count=0
    for line in que_file:
        data = json.loads(line)
        question = data['input']
        content = vector.query(question, EmbeddingModel=bgeEmbedding, k=3)
        rerank_content = reranker.rerank(question, content, k=2)
        # 最后选择最相似的文档, 交给LLM作为可参考上下文
        best_content = rerank_content[0]
        ans=remove_start_end(chat.chat(question, [], best_content))
        dicts["question"].append(question)
        dicts["answer"].append(ans)
        dicts["contexts"].append(best_content)
        dicts["reference"].append(data['output'])
        count= count+1
        print(count)
        current= {"question":question,"answer":ans,"contexts":best_content,"reference":data['output']}
        result_file.writelines(str(current))
    eval_data= Dataset.from_dict(dicts)
    result = Extra_evaluate(eval_data)
    print(result)
    df = result.to_pandas()
    df.to_csv("result/test_baseline_10_0907.csv",index=False)
    # result_file.write(result.ragas_traces)
           
# except Exception as e:
#     print("error!!!!",str(e))


