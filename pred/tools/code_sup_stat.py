import os
import json
from collections import defaultdict

# 自定义 bucket 间隔（例如 50、100）
bucket_size = 100

def get_id_bucket(_id: str, bucket_size: int) -> str:
    try:
        id_num = int(_id)
    except ValueError:
        return "invalid"
    
    if id_num < bucket_size:
        return f"<{bucket_size}"
    else:
        start = (id_num // bucket_size) * bucket_size
        end = start + bucket_size - 1
        return f"{start}-{end}"

# 路径设置
result_dir = "/home/ubuntu/Web/hzy/LooGLE-v2/results_code"

# 存储每个模型文件的统计结果
all_stats = {}

for filename in os.listdir(result_dir):
    if not filename.endswith(".jsonl"):
        continue

    file_path = os.path.join(result_dir, filename)
    bucket_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            _id = str(data.get("id", ""))
            judge = data.get("judge", False)

            bucket = get_id_bucket(_id, bucket_size)
            if bucket == "invalid":
                continue

            bucket_stats[bucket]["total"] += 1
            if judge is True:
                bucket_stats[bucket]["correct"] += 1

    all_stats[filename] = bucket_stats

# 打印输出
for filename, stats in all_stats.items():
    print(f"\nResults for: {filename} (bucket size = {bucket_size})")
    print(f"{'Bucket':<12}{'Correct':<10}{'Total':<10}{'Accuracy'}")
    print("-" * 45)
    def bucket_sort_key(x):
        return float('-1') if x.startswith("<") else int(x.split("-")[0])
    
    for bucket in sorted(stats.keys(), key=bucket_sort_key):
        s = stats[bucket]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"{bucket:<12}{s['correct']:<10}{s['total']:<10}{acc:.2%}")
