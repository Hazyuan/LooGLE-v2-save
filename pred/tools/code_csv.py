import os
import json
import pandas as pd
from tqdm import tqdm

# 指定结果目录
input_dir = "/home/ubuntu/Web/hzy/LooGLE-v2/results_code"

# 初始化：{id: {model_name: judge}}
result_dict = {}

# 遍历目录下所有 jsonl 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        model_name = filename.replace(".jsonl", "")
        file_path = os.path.join(input_dir, filename)

        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    qid = data.get("id")
                    judge = data.get("judge")  # True / False
                    if qid is None:
                        continue
                    if qid not in result_dict:
                        result_dict[qid] = {}
                    result_dict[qid][model_name] = judge
                except json.JSONDecodeError:
                    print(f"[Invalid JSON] {filename}")
                    continue

# 转换为 Pandas DataFrame
df = pd.DataFrame.from_dict(result_dict, orient="index")
df.index.name = "id"
df = df.sort_index(axis=1)  # 模型列排序（可选）

# 保存为 CSV 文件
output_path = "judge_matrix.csv"
df.to_csv(output_path)

print(f"✅ 保存成功: {output_path}")
