import os
import sys
import json
import math
from pathlib import Path

def parse_number(s):
    """从字符串中提取数值（去除$、M等），支持负号"""
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").replace("M", "").strip()
    try:
        return float(s)
    except:
        return None

def is_close(a, b, tol=0.05):
    """判断两个数值是否在误差5%以内，比较的是绝对值"""
    if a is None or b is None:
        return False
    a, b = abs(a), abs(b)
    if a == 0:
        return b < 1e-6
    return abs(a - b) / a <= tol

def process_file(file_path):
    updated_lines = []
    changed_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                modified = False

                if data.get("judge") is False and data.get("task") == "Metric Calculation":
                    correct = parse_number(data.get("correct_answer"))
                    pred = parse_number(data.get("pred_answer"))
                    if is_close(correct, pred):
                        print(f"[PASS] Line {idx} in {file_path.name}")
                        data["judge"] = True
                        modified = True
                        changed_count += 1
                    else:
                        print(f"[FAIL] Line {idx} in {file_path.name}")
                        print(f"  correct: {data.get('correct_answer')} | pred: {data.get('pred_answer')}")
                updated_lines.append(data)
            except json.JSONDecodeError:
                print(f"[ERROR] Invalid JSON on line {idx} in {file_path.name}")

    # 写回修改后的文件（覆盖原文件）
    with open(file_path, "w", encoding="utf-8") as f:
        for item in updated_lines:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nUpdated {changed_count} lines in {file_path.name}.")

def main(path_str):
    path = Path(path_str)
    if path.is_file():
        if path.suffix == ".jsonl":
            process_file(path)
    elif path.is_dir():
        for file in path.glob("*.jsonl"):
            process_file(file)
    else:
        print("Invalid path:", path_str)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python judge_and_update.py <jsonl file or directory>")
    else:
        main(sys.argv[1])
