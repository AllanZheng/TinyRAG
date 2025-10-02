
from evaluate.ragas_test import Extra_evaluate

import re
from datasets import Dataset
def remove_start_end(s):

    return re.sub(r'.*?</think>', '', s,flags=re.DOTALL)
txt_tables=[]
print("开始")
with open("result/test_base.json","r",encoding='utf-8') as result_file,open("test_data/ocean/test20_base_random.json", 'r',encoding="utf-8") as que_file:
    count=0
    line = result_file .readline()
    arr = str(line).split("{\'question\'")
    print(arr[2071])
    dicts = {}
    dicts["question"]=[]
    dicts["answer"]=[]
    dicts["contexts"]=[]
    dicts["reference"]=[]
    for i in range(1,2072):
        a = str(arr[i]).split("\'answer\'")
        dicts["question"].append(a[0])
        b = str(a[1]).split("\'contexts\'")
        dicts["answer"].append(b[0])
        c = str(b[1]).split("\'reference\'")
        dicts["contexts"].append([c[0]])
        dicts["reference"].append(c[1])
    print("2")
    eval_data= Dataset.from_dict(dicts)
    print("3")
    result = Extra_evaluate(eval_data)
    print(result)
    df = result.to_pandas()
    df.to_csv("result/test_baseline_0829.csv",index=False)
    # result_file.write(result.ragas_traces)