import random

import json

import pymysql
from pymysql.cursors import DictCursor
import configparser
import os
import pymysql
from pymysql.cursors import DictCursor
import configparser
import os
from collections import defaultdict
import dashscope
from dashscope import Generation
import os
from openai import OpenAI



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

    gpt_config = configparser.ConfigParser()
    gpt_config.read(config_file, encoding='utf-8')
    if 'gptkeys' not in gpt_config:
        raise KeyError("配置文件中缺少 [database] 节")
    gpt_config = {
        'deepseek': config['gptkeys'].get('deepseek', 'key'),
        'kimi': config['gptkeys'].get('kimi','key'),
        'qwen': config['gptkeys'].get('qwen', 'key'),

    }
    return db_config,gpt_config

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


import os
import requests


def generate_short_text(gpt_config,prompt_pre_comment,prompt_adstr, max_tokens=60):
    """
    调用 DeepSeek API 生成不超过 100 字的文本。

    Args:
        prompt (str): 用户要求
        max_tokens (int): 生成的最大 token 数（约等于字符数）
        api_key (str, optional): API 密钥，若不提供则从环境变量 DEEPSEEK_API_KEY 读取

    Returns:
        str: 生成的短文本，若失败返回空字符串
    """
    # if api_key is None:
    #     api_key = os.environ.get("DEEPSEEK_API_KEY")
    #     if not api_key:
    #         raise ValueError("未找到 API 密钥，请设置环境变量 DEEPSEEK_API_KEY 或直接传入 api_key")

    if 'deepseek' in gpt_config:
        api_key=gpt_config['deepseek']
    else:
        raise ValueError("未找到 API 密钥，请设置环境变量 DEEPSEEK_API_KEY 或直接传入 api_key")
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # 在系统提示中明确要求生成短文本
    system_prompt = "你是一个营销助手，请用简洁的语言生成不超过100字的营销话术。"
    prompt="这是用户的观点:"+prompt_pre_comment+", 结合这个观点,推荐下面的产品:"+prompt_adstr+" "
    payload = {
        "model": "deepseek-chat",  # 或 "deepseek-coder"
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"].strip()
        # 可选：强制截断到100字以内
        if len(text) > 100:
            text = text[:100]
        return text
    except Exception as e:
        print(f"调用 API 失败: {e}")
        return ""


def call_qwen3_5(gpt_config,system_prompt,prompt, max_tokens=150):
    if 'qwen' in gpt_config:
        api_key=gpt_config['qwen']
    else:
        raise ValueError("未找到 API 密钥，请设置环境变量 DEEPSEEK_API_KEY 或直接传入 api_key")

    try:
        client = OpenAI(
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
            messages=[
                # {'role': 'system', 'content': 'You are a helpful assistant.'},
                # {'role': 'user', 'content': '你是谁？'}
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
        )
        content=completion.choices[0].message.content
        print(content)

    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")
        return ''
    return content

# # ---------- 使用示例 ----------
# if __name__ == "__main__":
#     # 请确保已设置环境变量 DEEPSEEK_API_KEY
#     user_prompt = "用一句话介绍量子计算的核心优势，不超过100字。"
#     short_text = generate_short_text(user_prompt)
#     print(f"生成的文本（长度：{len(short_text)}字）：\n{short_text}")


def gen_data_for_sft():
    pass



def main():
    try:
        config_file = '../../../llm_data/config.txt'
        db_config,gpt_config = load_db_config(config_file)
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


    arr_for_sft_json=[]
    system_prompt = "你是一个营销助手，请用简洁的语言生成不超过100字的营销话术。"
    product = ['四级词汇app', '老友记口语app', '托福词汇app']

    k=0
    for record in top_records:
        if current_batch != record['phoneid']:
            current_batch = record['phoneid']
            print(f"\n批次 {current_batch}:")
        prompt_pre_comment=record['content']
        p_index=random.randint(0,len(product))
        if p_index>=len(product):
            p_index=len(product)-1
        prompt_adstr=product[p_index]
        # gpt_comment=generate_short_text(gpt_config, prompt_pre_comment, prompt_adstr, max_tokens=150)
        prompt = "这是用户的观点:" + prompt_pre_comment + ", 结合这个观点,推荐下面的产品:" + prompt_adstr + " "

        gpt_comment=call_qwen3_5(gpt_config,system_prompt, prompt, max_tokens=150)
        dat={"instruction": system_prompt, "input": prompt, "output": gpt_comment}
        arr_for_sft_json.append(dat)
        print("gpt_comment:"+gpt_comment)
        content_preview = record['content'][:100] + "..." if len(record['content']) > 100 else record['content']
        #print(f"  {record['rn']}. 好评数: {record['other']}  内容: {content_preview}")
        k=k+1
        if k>1200:
            break
    # with open("sft.json", "w", encoding="utf-8") as f:
    #     #json.dump(data, f, indent=4)  # indent 使格式美观
    #     json.dumps(arr_for_sft_json,f, indent=2)
    with open("sft.json", "w") as f:
        json_string = json.dumps(arr_for_sft_json, indent=2)  # Get JSON string
        f.write(json_string)




if __name__ == "__main__":
    main()



