import json
from pathlib import Path

def dedup_jsonl(input_path, output_path=None):
    input_path = Path(input_path)
    output_path = output_path or input_path.with_name(input_path.stem + "_dedup.jsonl")

    seen_ids = set()
    count_total = 0
    count_deduped = 0

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            count_total += 1
            try:
                item = json.loads(line)
                item_id = item.get("id")
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    json.dump(item, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    count_deduped += 1
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed line {count_total}")

    print(f"Total lines: {count_total}")
    print(f"Unique lines written: {count_deduped}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    dedup_jsonl("/home/ubuntu/Web/hzy/LooGLE-v2/results_evi3/DeepSeek-R1.jsonl")
