import psycopg2
import re
from psycopg2.extras import execute_values
# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",  # 替换为实际数据库名
    "user": "postgres",      # 替换为实际用户名
    "password": ""   # 替换为实际密码
}



def parse_triples_from_file(filename):
    """
    从txt文件中解析三元组数据
    格式：(subject, predicate, object)
    """
    triples = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
                
            # 使用正则表达式匹配三元组格式
            pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
            match = re.match(pattern, line)
            
            if match:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                obj = match.group(3).strip()
                key = subject
                source = "deepseek-32b"
                triples.append((subject, predicate, obj,key,source))
            else:
                print(f"警告: 第 {line_num} 行格式不正确: {line}")
    
    return triples

def import_to_postgres(triples, db_config,table_name="triples_1115"):
    """
    将三元组数据导入到PostgreSQL数据库
    """
    # insert_query = "INSERT INTO triples_1115 (start, mid, finish, key, source) VALUES %s"
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
 
        # 创建插入SQL
        insert_sql = f"""
        INSERT INTO {table_name} (start, mid, finish, key, source)
        VALUES %s
        """
        
        # 批量插入数据
        execute_values(cursor, insert_sql, triples)
        conn.commit()
        
        print(f"成功插入 {len(triples)} 条记录")
        
    except Exception as e:
        print(f"数据库操作失败: {e}")
        conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()
def main():

    # 输入文件路径
    input_file = 'D:\\Git\\SCI_RAG\\Kgs\\kgs1.txt'  # 替换为你的txt文件路径
    
    try:
        # 解析三元组数据
        print("正在解析文件...")
        triples = parse_triples_from_file(input_file)
        
        if not triples:
            print("未找到有效的三元组数据")
            return
        
        print(f"找到 {len(triples)} 个三元组")
        
        # 导入到数据库
        print("正在导入到数据库...")
        import_to_postgres(triples, DB_CONFIG)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()