import pymysql
from pymysql.cursors import DictCursor
import configparser
import os
import pymysql
from pymysql.cursors import DictCursor
import configparser
import os
from collections import defaultdict

def load_db_config(config_file='../../../llm_data/config.txt'):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 {config_file} 不存在")
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    if 'database' not in config:
        raise KeyError("配置文件中缺少 [database] 节")
    db_config = {
        'host': config['database'].get('host', 'localhost'),
        'user': config['database'].get('user'),
        'password': config['database'].get('password'),
        'database': config['database'].get('database'),
        'port': config['database'].getint('port', 3306),
        'charset': config['database'].get('charset', 'utf8mb4')
    }
    required = ['user', 'password', 'database']
    missing = [key for key in required if not db_config.get(key)]
    if missing:
        raise ValueError(f"配置文件中缺少必要参数: {', '.join(missing)}")
    return db_config

def get_all_valid_records(db_config):
    connection = None
    try:
        connection = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            port=db_config['port'],
            charset=db_config['charset'],
            cursorclass=DictCursor
        )
        sql = """
        SELECT phoneid, content, other
        FROM comments_for_train
        WHERE other IS NOT NULL AND other REGEXP '^[0-9]+$'
        """
        with connection.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
        return results
    except pymysql.MySQLError as e:
        print(f"数据库错误: {e}")
        return []
    finally:
        if connection:
            connection.close()

def get_top_two_by_batch(records):
    if not records:
        return []

    # 按批次分组
    batch_dict = defaultdict(list)
    for rec in records:
        # 确保 phoneid 不为空，若为空则用特殊值 "NULL_BATCH"
        batch_key = rec['phoneid'] if rec['phoneid'] is not None else "NULL_BATCH"
        # 验证 other 可转为整数
        try:
            int(rec['other'])
        except (ValueError, TypeError):
            continue
        batch_dict[batch_key].append(rec)

    top_records = []
    for batch_id, rec_list in batch_dict.items():
        # 按好评数降序排序
        rec_list.sort(key=lambda x: int(x['other']), reverse=True)
        # 取前两条
        taken = rec_list[:2]
        # 调试信息：打印该批次的总记录数和取到的条数
        print(f"批次 {batch_id}: 共有 {len(rec_list)} 条有效记录，取前 {len(taken)} 条")
        # 如果需要查看排序后的前几条，可以取消下面注释
        # for i, r in enumerate(rec_list[:5], 1):
        #     print(f"  排名{i}: 好评数={r['other']}, 内容={r['content'][:50]}...")
        for idx, rec in enumerate(taken, start=1):
            top_records.append({
                'phoneid': batch_id,
                'content': rec['content'],
                'other': rec['other'],
                'rn': idx
            })

    top_records.sort(key=lambda x: (x['phoneid'], x['rn']))
    return top_records


def gen_data_for_sft():
    pass

def main():
    try:
        db_config = load_db_config()
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"配置加载失败: {e}")
        return

    records = get_all_valid_records(db_config)
    if not records:
        print("未找到符合条件的记录。")
        return

    top_records = get_top_two_by_batch(records)
    if not top_records:
        print("处理后没有有效记录。")
        return

    print("\n最终输出结果（每个批次最多两条）:\n")
    current_batch = None
    for record in top_records:
        if current_batch != record['phoneid']:
            current_batch = record['phoneid']
            print(f"\n批次 {current_batch}:")
        content_preview = record['content'][:100] + "..." if len(record['content']) > 100 else record['content']
        print(f"  {record['rn']}. 好评数: {record['other']}  内容: {content_preview}")





if __name__ == "__main__":
    main()



