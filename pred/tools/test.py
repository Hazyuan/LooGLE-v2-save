from datasets import load_dataset
from pathlib import Path

# 指定数据集路径
dataset_path = "/home/ubuntu/Web/hzy/LooGLE-v2/datasets/LooGLE-v2"

# 加载 test 子集（HuggingFace datasets 会自动加载包含 "test" 的文件）
dataset = load_dataset(path=dataset_path, split="test")

# 查找特定 ID 的样本
target_id = "dcd0ea5b0f"
matched = dataset.filter(lambda x: x["id"] == target_id)

# 打印结果
if len(matched) > 0:
    item = matched[0]
    print("Evidence:\n", item.get("evidence"))
    print("\nQuestion:\n", item.get("question"))
    print("\nAnswer:\n", item.get("answer"))
else:
    print(f"ID '{target_id}' not found in test split.")
