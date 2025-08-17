import os
import re
import json
from datasets import load_dataset

# 加载数据
dataset_path = "/home/ubuntu/Web/hzy/LooGLE-v2/datasets/LooGLE-v2"
dataset = load_dataset(path=dataset_path)["test"]

# 正则定义
pattern_case = re.compile(r"<MASKED TARGET CASE>(.*?)<RELATED CASE>(.*)", re.DOTALL)
pattern_law = re.compile(r"<MASKED TARGET CASE>(.*?)<RELATED LAW>(.*)", re.DOTALL)
pattern_candidate = re.compile(r"<(CASE|LAW)_(\d+)>(.*?)(?=(<CASE_\d+>|<LAW_\d+>|$))", re.DOTALL)

# 存储结构化输出
results = []

for item in dataset:
    if item.get("source") != "Law":
        continue

    task = item.get("task")
    _id = str(item.get("id"))
    context = item.get("context", "")
    question = item.get("question", "")
    answer = item.get("answer", "")

    masked_section = ""
    candidate_section = ""
    matched = False

    if task == "Legal Case Retrieval":
        match = pattern_case.search(context)
        if match:
            masked_section = match.group(1).strip()
            candidate_section = match.group(2).strip()
            matched = True
    elif task == "Legal Article Extraction":
        match = pattern_law.search(context)
        if match:
            masked_section = match.group(1).strip()
            candidate_section = match.group(2).strip()
            matched = True

    if not matched:
        continue

    # 提取候选项
    candidates = []
    for match in pattern_candidate.finditer(candidate_section):
        label = f"{match.group(1)}_{match.group(2)}"
        text = match.group(3).strip()
        candidates.append({"id": label, "text": text})

    results.append({
        "id": _id,
        "task": task,
        "question": question,
        "answer": answer,
        "masked_target": masked_section,
        "candidate_pool": candidates
    })

# 保存为 jsonl
output_path = "/home/ubuntu/Web/hzy/LooGLE-v2/extracted_law_tasks.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(results)} items to {output_path}")
