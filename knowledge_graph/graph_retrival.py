import re
from typing import List, Tuple, Dict, Set
import jieba
import jieba.posseg as pseg

class KnowledgeGraphRetriever:
    def __init__(self):
        # 初始化知识图谱数据（示例数据）
        self.triples = self._init_knowledge_graph()
        # 实体别名映射
        self.entity_aliases = self._init_entity_aliases()
        
    def _init_knowledge_graph(self) -> List[Tuple]:
        """初始化示例知识图谱三元组"""
        triples = [
            # (主语, 关系, 宾语)
            ("苹果公司", "创始人", "史蒂夫·乔布斯"),
            ("苹果公司", "成立于", "1976年"),
            ("苹果公司", "总部位于", "加利福尼亚州"),
            ("苹果公司", "主要产品", "iPhone"),
            ("苹果公司", "主要产品", "MacBook"),
            ("史蒂夫·乔布斯", "出生日期", "1955年2月24日"),
            ("史蒂夫·乔布斯", "国籍", "美国"),
            ("iPhone", "发布时间", "2007年"),
            ("iPhone", "操作系统", "iOS"),
            ("MacBook", "类型", "笔记本电脑"),
            ("微软", "创始人", "比尔·盖茨"),
            ("微软", "成立于", "1975年"),
            ("谷歌", "创始人", "拉里·佩奇"),
        ]
        return triples
    
    def _init_entity_aliases(self) -> Dict[str, Set[str]]:
        """初始化实体别名映射"""
        aliases = {
            "苹果公司": {"苹果", "Apple", "Apple Inc."},
            "史蒂夫·乔布斯": {"乔布斯", "Steve Jobs"},
            "iPhone": {"苹果手机"},
            "MacBook": {"苹果笔记本", "Mac"},
            "微软": {"Microsoft"},
            "比尔·盖茨": {"盖茨", "Bill Gates"},
            "谷歌": {"Google", "谷歌公司"},
        }
        return aliases
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        从查询中提取实体
        这里使用简单的规则匹配，实际应用中可以使用NER模型
        """
        # 使用jieba进行分词和词性标注
        words = pseg.cut(query)
        
        entities = []
        # 提取可能的名词实体
        for word, flag in words:
            if flag in ['nr', 'ns', 'nt', 'nz']:  # 人名、地名、机构名、其他专有名词
                entities.append(word)
        
        # 同时尝试直接匹配知识图谱中的实体
        matched_entities = self._direct_entity_match(query)
        entities.extend(matched_entities)
        
        return list(set(entities))  # 去重
    
    def _direct_entity_match(self, query: str) -> List[str]:
        """直接匹配查询中的实体"""
        matched_entities = []
        
        # 匹配知识图谱中的标准实体
        all_entities = set()
        for subj, _, obj in self.triples:
            all_entities.add(subj)
            all_entities.add(obj)
        
        for entity in all_entities:
            if entity in query:
                matched_entities.append(entity)
        
        # 匹配实体别名
        for std_entity, aliases in self.entity_aliases.items():
            for alias in aliases:
                if alias in query:
                    matched_entities.append(std_entity)
        
        return matched_entities
    
    def normalize_entity(self, entity: str) -> str:
        """将实体别名转换为标准实体名称"""
        for std_entity, aliases in self.entity_aliases.items():
            if entity == std_entity or entity in aliases:
                return std_entity
        return entity
    
    def retrieve_by_entities(self, entities: List[str]) -> List[Tuple]:
        """根据实体检索相关的三元组"""
        results = []
        normalized_entities = [self.normalize_entity(e) for e in entities]
        
        for entity in normalized_entities:
            # 匹配包含该实体的所有三元组
            for triple in self.triples:
                if entity in triple:  # 实体可以是主语或宾语
                    results.append(triple)
        
        return list(set(results))  # 去重
    
    def search(self, query: str) -> Dict:
        """完整的检索流程"""
        print(f"查询: {query}")
        
        # 1. 实体识别
        entities = self.extract_entities_from_query(query)
        print(f"识别到的实体: {entities}")
        
        # 2. 实体标准化
        normalized_entities = [self.normalize_entity(e) for e in entities]
        print(f"标准化实体: {normalized_entities}")
        
        # 3. 三元组检索
        triples = self.retrieve_by_entities(normalized_entities)
        
        # 4. 结果组织
        result = {
            "query": query,
            "entities": normalized_entities,
            "triples": triples,
            "count": len(triples)
        }
        
        return result
    
    def pretty_print_results(self, results: Dict):
        """美化打印结果"""
        print("\n" + "="*50)
        print(f"查询: {results['query']}")
        print(f"识别实体: {', '.join(results['entities'])}")
        print(f"匹配到 {results['count']} 个三元组:")
        print("-" * 50)
        
        for i, (subj, rel, obj) in enumerate(results['triples'], 1):
            print(f"{i}. [{subj}] --{rel}--> [{obj}]")
        
        print("="*50 + "\n")

# 高级版本：使用相似度匹配
class AdvancedKGRetriever(KnowledgeGraphRetriever):
    def __init__(self):
        super().__init__()
        # 初始化实体向量（这里简化处理，实际可以使用词向量）
        self.entity_similarity_threshold = 0.6
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度（简化的Jaccard相似度）"""
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0
    
    def fuzzy_entity_match(self, query: str) -> List[str]:
        """模糊实体匹配"""
        # 收集所有实体
        all_entities = set()
        for subj, _, obj in self.triples:
            all_entities.add(subj)
            all_entities.add(obj)
        
        matched_entities = []
        query_terms = jieba.lcut(query)
        
        for term in query_terms:
            for entity in all_entities:
                similarity = self._calculate_similarity(term, entity)
                if similarity > self.entity_similarity_threshold:
                    matched_entities.append(entity)
        
        return matched_entities

# 使用示例
def main():
    # 初始化检索器
    retriever = KnowledgeGraphRetriever()
    
    # 测试查询
    test_queries = [
        "苹果公司的创始人是谁？",
        "乔布斯的个人信息",
        "iPhone是什么时候发布的？",
        "微软和谷歌的创始人",
        "苹果公司的主要产品有哪些？"
    ]
    
    print("基于知识图谱的三元组检索系统")
    print("=" * 50)
    
    for query in test_queries:
        results = retriever.search(query)
        retriever.pretty_print_results(results)
    
    # 测试高级版本
    print("\n" + "高级版本测试（模糊匹配）:")
    print("=" * 50)
    
    advanced_retriever = AdvancedKGRetriever()
    fuzzy_query = "苹果手机的信息"  # 包含别名
    results = advanced_retriever.search(fuzzy_query)
    advanced_retriever.pretty_print_results(results)

if __name__ == "__main__":
    # 安装依赖：pip install jieba
    main()