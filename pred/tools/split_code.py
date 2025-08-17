from datasets import load_dataset
import re
import json
from collections import defaultdict
import os

# 数据集路径
dataset_path = "/home/ubuntu/Web/hzy/LooGLE-v2/datasets/LooGLE-v2/"
dataset = load_dataset(path=dataset_path)
data = dataset["test"]

# 正则匹配规则
pattern_exact = re.compile(r"a call stack depth of (\d+)")
pattern_more_than = re.compile(r"stack depth of \*\*more than (\d+)\*\*")

# 分类存储
depth_to_ids = defaultdict(list)

# 遍历筛选
for item in data:
    if item.get("source") != "Code" or item.get("task") != "Call Graph Analysis":
        continue
    question = item.get("question", "")
    m1 = pattern_exact.search(question)
    m2 = pattern_more_than.search(question)
    if m1:
        depth = int(m1.group(1))
        depth_to_ids[depth].append(item["id"])
    elif m2:
        depth = int(m2.group(1))
        depth_to_ids[depth].append(item["id"])

# 保存到同目录下
output_file = os.path.join(dataset_path, "call_stack_depth_to_ids.json")
with open(output_file, "w") as f:
    json.dump({str(k): v for k, v in depth_to_ids.items()}, f, indent=2)
