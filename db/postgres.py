import json
import psycopg2
from psycopg2.extras import execute_values

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",  # 替换为实际数据库名
    "user": "postgres",      # 替换为实际用户名
    "password": ""   # 替换为实际密码
}

# 读取并解析txt文件中的JSON数据
def read_json_from_txt(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                # 解析每行JSON数据
                json_obj = json.loads(line)
                data_type = "test"
                # 验证必需字段
                if all(key in json_obj for key in ['task_type', 'input', 'output']):
                    data.append((
                        json_obj['task_type'],
                        json_obj['input'],
                        json_obj['output'],
                        "train"
                    ))
                else:
                    print(f"第{line_num}行缺少必需字段，已跳过")
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
    return data

# 插入数据到PostgreSQL
def insert_into_postgres(data, table_name='your_table'):  # 替换为实际表名
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
 
        # 创建插入SQL
        insert_sql = f"""
        INSERT INTO {table_name} (task_type, input, output,data_type)
        VALUES %s
        """
        
        # 批量插入数据
        execute_values(cursor, insert_sql, data)
        conn.commit()
        
        print(f"成功插入 {len(data)} 条记录")
        
    except Exception as e:
        print(f"数据库操作失败: {e}")
        conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    # 配置参数
    TXT_FILE = "D:\\Git\\TinyRAG\\train_data\ocean\\train80_base_random.txt"  # 替换为你的txt文件路径
    TABLE_NAME = "ocean_bench"  # 替换为目标表名
    
    # 执行数据导入
    json_data = read_json_from_txt(TXT_FILE)
    if json_data:
        insert_into_postgres(json_data, TABLE_NAME)
    else:
        print("未找到有效数据")