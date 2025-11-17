import json
import psycopg2
import os
from psycopg2.extras import execute_values
from typing import Dict, List, Any

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",  # 替换为实际数据库名
    "user": "postgres",      # 替换为实际用户名
    "password": ""   # 替换为实际密码
}


# 创建表结构（根据HotpotQA数据集字段）
def create_table_schema(table_name: str = "hotpotqa"):
    """创建HotpotQA数据表"""
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        _id TEXT,
        question TEXT,
        answer TEXT,
        type TEXT,
        level TEXT,
        supporting_facts JSONB,
        context JSONB,
        difficulty TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- 创建索引以提高查询性能
    CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}(_id);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_type ON {table_name}(type);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_level ON {table_name}(level);
    CREATE INDEX IF NOT EXISTS idx_{table_name}_difficulty ON {table_name}(difficulty);
    """
    
    return create_table_sql

# 读取HotpotQA数据集
def read_hotpotqa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    读取HotpotQA数据集文件
    支持格式：
    - 单个JSON对象（包含数据数组）
    - JSON Lines格式（每行一个JSON对象）
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 尝试解析为单个JSON文件
            try:
                content = json.load(file)
                # 如果是字典且包含数据字段
                if isinstance(content, dict):
                    # 查找常见的数据字段名
                    for key in ['data', 'questions', 'rows', 'documents']:
                        if key in content:
                            data = content[key]
                            break
                    # 如果没有找到特定字段，使用整个内容
                    if not data:
                        data = [content]
                # 如果是列表，直接使用
                elif isinstance(content, list):
                    data = content
                    
            # 如果是JSON Lines格式
            except json.JSONDecodeError:
                file.seek(0)  # 回到文件开头
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"第{line_num}行JSON解析错误: {e}")
                        continue
                        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        
    print(f"成功读取 {len(data)} 条记录")
    return data

# 数据清洗和转换
def transform_hotpotqa_data(raw_data: List[Dict[str, Any]]) -> List[tuple]:
    """转换数据为数据库插入格式"""
    transformed = []
    
    for item in raw_data:
        try:
            # 提取常见字段（根据实际数据集结构调整）
            record = (
                item.get('_id'),                    # 原始ID
                item.get('question', ''),           # 问题
                item.get('answer', ''),             # 答案
                item.get('type', ''),             # 类型
                item.get('level', ''),             # 级别
                json.dumps(item.get('supporting_facts', [])),  # 支持事实
                json.dumps(item.get('context', {})),           # 上下文
                item.get('difficulty', '')         # 难度
            )
            transformed.append(record)
            
        except Exception as e:
            print(f"转换数据时出错: {e}, 数据: {item}")
            continue
            
    return transformed

# 插入数据到PostgreSQL
def insert_hotpotqa_data(data: List[tuple], table_name: str = "hotpotqa"):
    """将数据插入到PostgreSQL数据库"""
    
    if not data:
        print("没有数据可插入")
        return
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute(create_table_schema(table_name))
        
        # 插入数据的SQL
        insert_sql = f"""
        INSERT INTO {table_name} (_id, question, answer, type, level, supporting_facts, context, difficulty)
        VALUES %s
        """
        
        # 批量插入
        execute_values(cursor, insert_sql, data)
        conn.commit()
        
        print(f"成功插入 {len(data)} 条记录到表 {table_name}")
        
    except Exception as e:
        print(f"数据库操作失败: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()

# 主函数
def main():
    # 配置参数
    for num in range(1,12):
        DATA_FILE = f"D:\\Git\\SCI_RAG\\data\\hotpot_{num}.json"  # 替换为实际文件路径
        TABLE_NAME = "hotpotqa"  # 表名
        
        print(f"{num}开始导入HotpotQA数据集...")
        
        # 读取数据
        print("读取数据文件中...")
        raw_data = read_hotpotqa_data(DATA_FILE)
        
        if not raw_data:
            print(f"{num}未读取到数据，请检查文件路径和格式")
            return
        
        # 转换数据
        print(f"{num}转换数据格式...")
        transformed_data = transform_hotpotqa_data(raw_data)
        
        if not transformed_data:
            print(f"{num}数据转换失败")
            return
        
        # 插入数据库
        print(f"{num}插入数据到PostgreSQL...")
        insert_hotpotqa_data(transformed_data, TABLE_NAME)
        
        print(f"{num}数据导入完成！")

if __name__ == "__main__":
    main()