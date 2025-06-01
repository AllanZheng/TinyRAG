#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json 
from typing import Dict, List, Optional, Tuple, Union
import time
API_BASE = "http://172.18.127.124:40011/v1"
API_KEY = "ollama"  # vllm服务默认不需要验证，但需要设置一个虚拟key
# MODEL_NAME = "deepseek-R1:1.5B"
PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""Use the context to answer the user's questions. If you don't know the answer, say you don't know. Always respond in English.
        Question: {question}
        Relevant context:
        ···
        {context}
        ···
        If the provided context does not allow you to formulate an answer, respond that the database does not contain this information and that you don't know。
        Helpful response:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中英文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str,config:dict) -> str:
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "qwq-32b") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            base_url= API_BASE,
            api_key=API_KEY
        )
        # client.api_key = API_KEY
        # client.base_url = API_BASE
        # client.models =self.model
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        print("Start")
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=history,
                max_tokens=5000,
                stream=False,
                temperature=0.1
            )
            end_time = time.time()
            print("耗费时间:",end_time-start_time)
        except:
            print("请求失败: {str(e)}")
        return response.choices[0].message.content

class InternLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str='') -> str:
        prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response


    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda()

class DashscopeChat(BaseModel):
    def __init__(self, path: str = '', model: str = "qwen-turbo") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[Dict], content: str) -> str:
        import dashscope
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = dashscope.Generation.call(
            model=self.model,
            messages=history,
            result_format='message',
            max_tokens=150,
            temperature=0.1
        )
        return response.output.choices[0].message.content
    

class ZhipuChat(BaseModel):
    def __init__(self, path: str = '', model: str = "glm-4") -> None:
        super().__init__(path)
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        self.model = model

    def chat(self, prompt: str, history: List[Dict], content: str) -> str:
        history.append({'role': 'user', 'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message