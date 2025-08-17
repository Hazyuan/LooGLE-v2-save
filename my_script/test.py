import json

input_path = "/home/ubuntu/Web/hzy/LooGLE-v2/results/glm-4-9b-chat-1m.jsonl"
output_path = "/home/ubuntu/Web/hzy/LooGLE-v2/results/glm-4-9b-chat-1m_dedup.jsonl"

seen_ids = set()
deduped_lines = []

with open(input_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        data = json.loads(line)
        uid = data.get("id")
        if uid not in seen_ids:
            seen_ids.add(uid)
            deduped_lines.append(data)

# 保存去重后的结果（可选）
with open(output_path, 'w', encoding='utf-8') as outfile:
    for item in deduped_lines:
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"去重后共保留 {len(deduped_lines)} 条记录。")
