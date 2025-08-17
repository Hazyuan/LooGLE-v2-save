import json
import re
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def extract_query_from_mask(masked_target, mask_id, window=100):
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
output_file = input_file.replace(".jsonl", "_bm25_tfidf_result.jsonl")

results = []
bm25_correct = 0
tfidf_correct = 0
total = 0

# 每个 task 的统计记录
bm25_task_stats = defaultdict(lambda: {"correct": 0, "total": 0})
tfidf_task_stats = defaultdict(lambda: {"correct": 0, "total": 0})

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Evaluating"):
        item = json.loads(line)
        question = item["question"]
        answer = item["answer"]
        masked_target = item["masked_target"]
        candidate_pool = item["candidate_pool"]
        task = item.get("task", "Unknown")

        mask_match = re.search(r"<(MASK_\d+)>", question)
        if not mask_match:
            continue

        mask_id = mask_match.group(1)
        query_text = extract_query_from_mask(masked_target, mask_id)
        query_tokens = query_text.split()

        corpus_texts = [c["text"] for c in candidate_pool]
        candidate_ids = [c["id"] for c in candidate_pool]

        ### BM25 baseline ###
        tokenized_corpus = [doc.split() for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_index = bm25_scores.argmax()
        bm25_predicted_id = candidate_ids[bm25_top_index]
        bm25_is_correct = (bm25_predicted_id == answer.strip())
        bm25_correct += int(bm25_is_correct)
        bm25_task_stats[task]["correct"] += int(bm25_is_correct)

        ### TF-IDF baseline ###
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus_texts + [query_text])
        query_vec = tfidf_matrix[-1]
        doc_matrix = tfidf_matrix[:-1]
        sims = cosine_similarity(query_vec, doc_matrix)[0]
        tfidf_top_index = sims.argmax()
        tfidf_predicted_id = candidate_ids[tfidf_top_index]
        tfidf_is_correct = (tfidf_predicted_id == answer.strip())
        tfidf_correct += int(tfidf_is_correct)
        tfidf_task_stats[task]["correct"] += int(tfidf_is_correct)

        ### 保存结果 ###
        total += 1
        bm25_task_stats[task]["total"] += 1
        tfidf_task_stats[task]["total"] += 1
        item["query"] = query_text
        item["bm25_top1"] = bm25_predicted_id
        item["bm25_correct"] = bm25_is_correct
        item["tfidf_top1"] = tfidf_predicted_id
        item["tfidf_correct"] = tfidf_is_correct
        item.pop("candidate_pool", None)
        item.pop("masked_target", None)
        results.append(item)

# 输出准确率
print(f"\nBM25 Overall Accuracy: {bm25_correct}/{total} = {bm25_correct / total:.2%}")
print(f"TF-IDF Overall Accuracy: {tfidf_correct}/{total} = {tfidf_correct / total:.2%}")

# 按 task 输出
def print_stats(title, stats):
    print(f"\n{title}")
    print(f"{'Task':<30} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print("-" * 60)
    for task, stat in stats.items():
        acc = stat["correct"] / stat["total"] if stat["total"] > 0 else 0.0
        print(f"{task:<30} {stat['correct']:<10} {stat['total']:<10} {acc:.2%}")

print_stats("BM25 Accuracy by Task:", bm25_task_stats)
print_stats("TF-IDF Accuracy by Task:", tfidf_task_stats)

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Saved results to: {output_file}")
