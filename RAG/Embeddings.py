#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   Embeddings.py
@Time    :   2024/02/10 21:55:39
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import requests
os.environ['CURL_CA_BUNDLE'] = ''
from dotenv import load_dotenv, find_dotenv
from sklearn.covariance import EmpiricalCovariance
_ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
        self.cov_estimator = EmpiricalCovariance(assume_centered=True)

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    def get_np_embeddings(self, texts)-> List[float]:
        raise NotImplementedError
    
    def fit_covariance(self, embeddings):
        raise NotImplementedError
    #     """拟合协方差矩阵（考虑中心化）
    #     Args:
    #         embeddings: 文本向量矩阵
    #     """
    #     self.cov_estimator.fit(embeddings)
    #     # 获取协方差矩阵的伪逆（避免奇异矩阵问题）
    #     try:
    #         self.inv_cov = np.linalg.inv(self.cov_estimator.covariance_)
    #     except np.linalg.LinAlgError:
    #         self.inv_cov = np.linalg.pinv(self.cov_estimator.covariance_)

    def mahalanobis_distance(self, vec1, vec2):
        raise NotImplementedError
    #     """计算马氏距离（基于协方差矩阵）
    #     Args:
    #         vec1, vec2: 两个向量
    #     Returns:
    #         马氏距离值
    #     """
    #     diff = vec1 - vec2
    #     return np.sqrt(diff.dot(self.inv_cov).dot(diff))
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    # @classmethod
    def covariance_similarity(cls,vector1: List[float], vector2: List[float]) -> float:
        raise NotImplementedError
    #     """
    #     calculate cosine similarity between two vectors
    #     """
    #     # dot_product = np.dot(vector1, vector2)
    #     ins = cls() 
    #     ins.fit_covariance(vector1+vector2)
    #     dist = ins.mahalanobis_distance(vector1, vector2)
    #         # 将距离转换为相似度（0-1范围)
        
    #     return 1 / (1 + dist)
    def load_background_data(self, texts: List[str]):
        raise NotImplementedError
class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError

class JinaEmbedding(BaseEmbeddings):
    """
    class for Jina embeddings
    """
    def __init__(self, path: str = 'jinaai/jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()

    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()

    def load_model(self):
        import torch
        from transformers import AutoModel
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
        return model

class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
        model="embedding-2",
        input=text,
        )
        return response.data[0].embedding

class DashscopeEmbedding(BaseEmbeddings):
    """
    class for Dashscope embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            self.client = dashscope.TextEmbedding

    def get_embedding(self, text: str, model: str='text-embedding-v1') -> List[float]:
        response = self.client.call(
            model=model,
            input=text
        )
        return response.output['embeddings'][0]['embedding']


class BgeEmbedding(BaseEmbeddings):
    """
    class for BGE embeddings
    """

    def __init__(self, path: str = 'BAAI/bge-base-en-v1.5', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model, self._tokenizer = self.load_model(path)
        self.model =self._model
        # self.cov_estimator = EmpiricalCovariance(assume_centered=True)
        self.background_vectors = None
        self.inv_cov_matrix = None
    def get_embedding(self, text: str) -> List[float]:
        import torch
        encoded_input = self._tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self._model.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings[0].tolist()
    def load_model(self, path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(device)
        model.eval()
        return model, tokenizer
    # def get_np_embeddings(self, texts):
    #     """生成文本的稠密向量表示
    #     Args:
    #         texts: 文本列表
    #     Returns:
    #         (n_samples, dim)维度的numpy数组s
    #     """
    #     return np.array(self._tokenizer.encode(texts)['dense_vecs'])
    
    def load_background_data(self, vectors: List[float]):
        """
        加载背景数据集以计算协方差矩阵
        Args:
            texts: 用于计算协方差的背景文本列表
        """
        # background_vectors = []
        # for text in texts:
        #     vec = self.get_embedding(text)
        #     background_vectors.append(vec)
        
        self.background_vectors = np.array(vectors)
        self.fit_covariance()
    def fit_covariance(self):
        """
        计算背景数据的协方差矩阵及其逆矩阵
        """
        if self.background_vectors is None or len(self.background_vectors) < 2:
            raise ValueError("需要至少2个背景数据点来计算协方差")
        
        # 使用EmpiricalCovariance计算协方差
        cov_estimator = EmpiricalCovariance(assume_centered=False)
        cov_estimator.fit(self.background_vectors)
        cov_matrix = cov_estimator.covariance_
        
        # 添加小的正则项以避免奇异性
        cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
        
        try:
            self.inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            self.inv_cov_matrix = np.linalg.pinv(cov_matrix)
        # 获取协方差矩阵的伪逆（避免奇异矩阵问题）
        # try:
        #     self.inv_cov = np.linalg.inv(self.cov_estimator.covariance_)
        # except np.linalg.LinAlgError:
        #     self.inv_cov = np.linalg.pinv(self.cov_estimator.covariance_)

    def mahalanobis_distance(self, vec1, vec2):
        """计算马氏距离（基于协方差矩阵）
        Args:
            vec1, vec2: 两个向量
        Returns:
            马氏距离值
        """        
        embedding1 = np.array(vec1)
        embedding2 = np.array(vec2)
        diff = embedding1 - embedding2
        return  np.sqrt(np.dot(np.dot(diff, self.inv_cov_matrix), diff))
    def covariance_similarity(self,vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        # dot_product = np.dot(vector1, vector2)
        # self.fit_covariance(vector1+vector2)
        dist = self.mahalanobis_distance(vector1, vector2)
            # 将距离转换为相似度（0-1范围)
        print(1 / (1 + dist),dist)
        return 1 / (1 + dist)
        # return dist
class RemoteBgeEmbedding(BaseEmbeddings):
    """
    class for BGE embeddings
    """

    def __init__(self, path: str = 'BAAI/bge-base-zh-v1.5', is_api: bool = False) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, texts: str) -> List[float]:
       
        url = "http://172.18.127.124:40697/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        payload = {
        "input": texts,
        "model": "bge-m3",
        "encoding_type": "dense"  # 可选dense/sparse/dense_sparse
        }
    
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(response.json())
            return response.json()["data"][0]["embedding"]  # 首条文本的向量
        else:
            raise Exception(f"API调用失败: {response.text}")

    def load_model(self, path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(device)
        model.eval()
        return model, tokenizer