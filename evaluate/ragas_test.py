from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import typing as t
from datasets import Dataset
llm_api_key = "gpustack_dee9ca823290886c_5edfc86aeeeceb1e9ee5162941cb2cb5"
llm_base_url = "http://172.18.30.122:40017/v1"
embd_api_key = "gpustack_dee9ca823290886c_5edfc86aeeeceb1e9ee5162941cb2cb5"
embd_base_url = "http://172.18.30.122:40050/v1"

# generator with openai models
generator_llm = ChatOpenAI(model="Qwen3-32B", api_key=llm_api_key, base_url=llm_base_url,timeout=30)

embeddings = OpenAIEmbeddings(model="bge-m3", api_key=embd_api_key, base_url=embd_base_url,timeout=15)
lang_llm=LangchainLLMWrapper(generator_llm) 
lang_embeddings= LangchainEmbeddingsWrapper(embeddings)
from ragas.metrics import (
    context_precision, 
    faithfulness, 
    answer_relevancy, 
    context_recall
)

# default_dataset = Dataset.from_dict({
#     "question": ["爱因斯坦出生在哪里？"],
#     "answer": ["爱因斯坦于1879年3月14日出生在德国。"],
#     "contexts": [["阿尔伯特·爱因斯坦（生于1879年3月14日）是德国出生的物理学家。"]],
#     "reference": ["爱因斯坦出生于德国。"]
# })
q="爱因斯坦出生在哪里？"
a="爱因斯坦于1879年3月14日出生在德国。"
c=["阿尔伯特·爱因斯坦（生于1879年3月14日）是德国出生的物理学家。"]
r="爱因斯坦出生于德国。"
q1,a1,c1,r1=[],[],[],[]
for i in range(0,5):
    q1.append(q)
    a1.append(a)
    c1.append(c)
    r1.append(r)
default_dataset = Dataset.from_dict({
    "question": q1,
    "answer": a1,
    "contexts": c1,
    "reference": r1
})
# 选择评测指标
# metrics = [context_precision, faithfulness, answer_relevancy, context_recall]
default_metrics = [ context_precision,faithfulness, answer_relevancy, context_recall]

# faithfulness.llm = lang_llm
# answer_relevancy.llm = lang_llm
# answer_relevancy.embeddings = embeddings
# context_recall.llm = generator_llm
# context_precision.llm = generator_llm

# 运行评估
def Extra_evaluate(
        dataset:t.Optional[Dataset.from_dict]=default_dataset, 
        metric:t.Optional[list]=default_metrics,
        cust_llm:t.Optional[LangchainLLMWrapper]=lang_llm,
        embedding:t.Optional[LangchainEmbeddingsWrapper]=lang_embeddings,
        )->EvaluationResult:

    result = evaluate(dataset, metrics=metric,llm=cust_llm,embeddings=embedding)
    df = result.to_pandas()
    print(df.head())
    return result
# print(Extra_evaluate(default_dataset))