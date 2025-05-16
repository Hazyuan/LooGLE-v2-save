import json
import argparse
from collections import defaultdict

def evaluate_accuracy_by_group(file_path):
    group_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    overall_total = 0
    overall_correct = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            source = data.get("source", "Unknown")
            task = data.get("task", "Unknown")
            judge = data.get("judge", None)

            if isinstance(judge, bool):
                key = (source, task)
                group_stats[key]["total"] += 1
                overall_total += 1
                if judge:
                    group_stats[key]["correct"] += 1
                    overall_correct += 1
            elif isinstance(judge, float):
                key = (source, task)
                group_stats[key]["total"] += 1
                overall_total += 1
                group_stats[key]["correct"] += judge/100
                overall_correct += judge/100
    print("\n[分组准确率统计]")
    for (source, task), stats in sorted(group_stats.items()):
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"Source: {source:<10} | Task: {task:<30} | Accuracy: {accuracy:.2f}% ({correct}/{total})")

    overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print(f"\n[总体准确率]")
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate grouped accuracy from JSONL results.")
    parser.add_argument("--input_path", type=str, default=None, help="Path to result JSONL file.")

    args = parser.parse_args()


    print(f"[INFO] Evaluating file: {args.input_path}")
    evaluate_accuracy_by_group(args.input_path)
