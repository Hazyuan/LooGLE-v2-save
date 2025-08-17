    import json
    import re
    from collections import defaultdict
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm

    def extract_query_from_mask(masked_target, mask_id, window=50):
        """从 masked_target 中提取包含 <MASK_x> 的上下文窗口"""
        pattern = re.compile(re.escape(f"<{mask_id}>"))
        matches = list(pattern.finditer(masked_target))
        queries = []

        for m in matches:
            start = max(0, m.start() - window)
            end = min(len(masked_target), m.end() + window)
            snippet = masked_target[start:end]
            queries.append(snippet.strip())

        return " ".join(queries)

    # 输入路径
    input_file = "/home/ubuntu/Web/hzy/LooGLE-v2/extracted_law_tasks.jsonl"
    output_file = input_file.replace(".jsonl", "_bm25_result.jsonl")

    results = []
    correct = 0
    total = 0

    # 每个 task 的统计记录
    task_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            item = json.loads(line)
            question = item["question"]
            answer = item["answer"]
            masked_target = item["masked_target"]
            candidate_pool = item["candidate_pool"]
            task = item.get("task", "Unknown")

            # 提取 <MASK_x>
            mask_match = re.search(r"<(MASK_\d+)>", question)
            if not mask_match:
                continue

            mask_id = mask_match.group(1)
            query_text = extract_query_from_mask(masked_target, mask_id)
            query_tokens = query_text.split()

            corpus = [c["text"].split() for c in candidate_pool]
            candidate_ids = [c["id"] for c in candidate_pool]

            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
            top_index = scores.argmax()
            predicted_id = candidate_ids[top_index]

            is_correct = (predicted_id == answer.strip())
            correct += int(is_correct)
            total += 1

            task_stats[task]["correct"] += int(is_correct)
            task_stats[task]["total"] += 1
            item["query"] = query_text
            item["bm25_top1"] = predicted_id
            item["bm25_correct"] = is_correct
            item.pop("candidate_pool", None)
            item.pop("masked_target", None)
            results.append(item)

    # 输出总体准确率
    print(f"\nBM25 Overall Accuracy: {correct}/{total} = {correct / total:.2%}")

    # 按 task 输出统计
    print("\nAccuracy by Task:")
    print(f"{'Task':<30} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print("-" * 60)
    for task, stat in task_stats.items():
        acc = stat["correct"] / stat["total"] if stat["total"] > 0 else 0.0
        print(f"{task:<30} {stat['correct']:<10} {stat['total']:<10} {acc:.2%}")

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved results to: {output_file}")
