import json
import argparse

def remove_null_pred_answer(input_path, output_path):
    kept = 0
    removed = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("pred_answer") is not None:
                json.dump(obj, outfile, ensure_ascii=False)
                outfile.write('\n')
                kept += 1
            else:
                removed += 1

    print(f"已完成清理：保留 {kept} 条记录，移除 {removed} 条 pred_answer 为 null 的记录")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="移除 pred_answer 为 null 的 JSONL 行")
    parser.add_argument('--input', required=True, help='输入的 JSONL 文件路径')
    parser.add_argument('--output', required=True, help='输出的 JSONL 文件路径（已过滤）')

    args = parser.parse_args()
    remove_null_pred_answer(args.input, args.output)
