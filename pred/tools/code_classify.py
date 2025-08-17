import json
import argparse
import os

def load_depth_mapping(depth_file):
    with open(depth_file, 'r') as f:
        return json.load(f)

def load_judgments(result_file):
    id_to_judge = {}
    with open(result_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            _id = data.get("id")
            judge = data.get("judge")
            if _id is not None and isinstance(judge, bool):
                id_to_judge[_id] = judge
    return id_to_judge

def evaluate_by_depth(result_file, depth_to_ids):
    id_to_judge = load_judgments(result_file)
    result = {}
    for depth, ids in sorted(depth_to_ids.items(), key=lambda x: int(x[0])):
        total = 0
        correct = 0
        for _id in ids:
            if _id in id_to_judge:
                total += 1
                if id_to_judge[_id]:
                    correct += 1
        acc = correct / total if total > 0 else 0.0
        result[depth] = {"total": total, "correct": correct, "acc": acc}
    return result

def print_result_table(model_name, stat_dict):
    print(f"\nModel: {model_name}")
    print(f"{'Depth':<10}{'Total':<10}{'Correct':<10}{'Accuracy'}")
    print("-" * 40)
    for depth in sorted(stat_dict, key=lambda x: int(x)):
        stats = stat_dict[depth]
        print(f"{depth:<10}{stats['total']:<10}{stats['correct']:<10}{stats['acc']:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to result file (.jsonl) or directory")
    args = parser.parse_args()

    depth_file = "/home/ubuntu/Web/hzy/LooGLE-v2/datasets/LooGLE-v2/call_stack_depth_to_ids.json"
    depth_to_ids = load_depth_mapping(depth_file)

    if os.path.isdir(args.path):
        for filename in sorted(os.listdir(args.path)):
            if filename.endswith(".jsonl"):
                full_path = os.path.join(args.path, filename)
                model_name = os.path.splitext(filename)[0]
                stats = evaluate_by_depth(full_path, depth_to_ids)
                print_result_table(model_name, stats)
    else:
        model_name = os.path.splitext(os.path.basename(args.path))[0]
        stats = evaluate_by_depth(args.path, depth_to_ids)
        print_result_table(model_name, stats)
