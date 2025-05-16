import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
from collections import OrderedDict
from datetime import datetime
import requests
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import ast


def truncate(model, prompt, tokenizer, max_len):
    if "gpt" in model or "o1" in model or "o3" in model or "o4" in model:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
    else:
        input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len // 2] + input_ids[-max_len // 2:]
        if "gpt" in model or "o1" in model or "o3" in model or "o4" in model or model == "Fin-R1":
            prompt = tokenizer.decode(input_ids)
        else:
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt


def format_prompt(model, data, tokenizer, max_len, with_context):
    if with_context == False:
        data["context"] = "I would not provide you with the context. Please choose the most likely option based on your knowledge and intuition."
    prompt = data["instruction"].format(context=data["context"], options=data["options"], question=data["question"])
    prompt = truncate(model, prompt, tokenizer, max_len)
    return prompt


def multiple_choice_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is: \(([A-D])', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is: ([A-D])', response)
    if match:
        return match.group(1)
    else:
        return None



def extract_case_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is\s*\(CASE_\d+', response)
    if match:
        return re.search(r'CASE_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is:\s*\(CASE_\d+', response)
    if match:
        return re.search(r'CASE_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is?\s*CASE_\d+', response)
    if match:
        return re.search(r'CASE_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is:?\s*CASE_\d+', response)
    if match:
        return re.search(r'CASE_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is\s*<CASE_\d+', response)
    if match:
        return re.search(r'CASE_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is:\s*<CASE_\d+', response)
    if match:
        return re.search(r'CASE_\d+', match.group(0)).group(0)
    return None



def extract_law_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is\s*\(LAW_\d+', response)
    if match:
        return re.search(r'LAW_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is:\s*\(LAW_\d+', response)
    if match:
        return re.search(r'LAW_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is?\s*LAW_\d+', response)
    if match:
        return re.search(r'LAW_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is:?\s*LAW_\d+', response)
    if match:
        return re.search(r'LAW_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is\s*<LAW_\d+', response)
    if match:
        return re.search(r'LAW_\d+', match.group(0)).group(0)
    match = re.search(r'The correct answer is:\s*<LAW_\d+', response)
    if match:
        return re.search(r'LAW_\d+', match.group(0)).group(0)
    return None

def extract_qa_answer_trend_analysis(response):
    match = re.search(r"[Tt]he correct answer is+(.*)", response)
    if not match:
        return []
    answer_text = match.group(1)
    pattern = re.compile(r'\b\d{4}-\d{4}\b')
    answer_text = pattern.findall(answer_text)
    return answer_text


def extract_qa_answer(response):
    match = re.search(r"[Tt]he correct answer is+(.*)", response)
    if not match:
        return []
    answer_text = match.group(1)
    pattern = re.compile(r'\d{1,6}(?:,\d{3})*(?:\.\d+)?(?:[M%])?')
    answer_text = pattern.findall(answer_text)
    return answer_text

def extract_version_control_answer(response):
    matches = re.findall(r'[\w/]+\.py', response)
    return list(set(matches))


def compute_jaccard_score(list1, list2):

    if not list1 or not list2:
        return 0.0
    mlb = MultiLabelBinarizer()
    binarized = mlb.fit_transform([list1, list2])
    score = jaccard_score(binarized[0], binarized[1]) * 100
    return round(score, 2)

def normalize_number(s):
    if isinstance(s, list):
        s = s[0] if s else ''
    if not s:
        return None
    try:
        s = s.replace(',', '').replace('$', '').strip()
        multiplier = 1.0
        if s.endswith('M'):
            s = s[:-1]
        elif s.endswith('%'):
            s = s[:-1]
            multiplier = 1
        return float(s) * multiplier
    except ValueError:
        return None


def load_model_config(config_path, target_model_name):
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            model_info = json.loads(line.strip())
            if model_info.get("name") == target_model_name:
                return {
                    "model": model_info.get("model", ""),
                    "max_len": model_info.get("max_len", ""),
                    "BASE_URL": model_info.get("base_url", ""),
                    "API_KEY": model_info.get("api_key", "")
                }
    raise ValueError(f"Model name '{target_model_name}' not found in {config_path}")

def query_llm(data, model, max_len, tokenizer, client, with_context, temperature=0.1, max_new_tokens=32, stop=None):
    # truncate
    prompt = format_prompt(model, data, tokenizer, max_len, with_context)
    tries = 0
    while tries < 5:
        tries += 1
        try:
            if "Kimi" in model:
                url = str(client.base_url) + "chat/completions/"
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature
                }
                response = requests.post(url, headers=headers, json=data)
                return response.json()["choices"][0]["message"]["content"]
            if "o1" in model or "o3" in model or "o4" in model:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                )
                print(response.usage)
                return response.output_text
            elif "gpt" in model:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature = temperature,
                )
                print(response.usage)
                return response.output_text
            elif "Yarn-Mistral" in model or "Yi" in model:
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return response.choices[0].text
            else:
                if model == "deepseek-ai/DeepSeek-V3":
                    model = "deepseek-chat"
                elif model == "deepseek-ai/DeepSeek-R1":
                    model = "deepseek-reasoner"
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    stream=False,
                )
                return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def get_pred(dataset, args, fout):
    config = load_model_config("config/models.jsonl", args.model)
    model = config["model"]
    max_len = config["max_len"]
    BASE_URL = config["BASE_URL"]
    API_KEY = config["API_KEY"]
    if "gpt" in model or "o1" in model or "o3" in model or "o4" in model or "Fin-R1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    for data in tqdm(dataset):
        output = query_llm(data, model, max_len, tokenizer, client, args.with_context, temperature=0.1, max_new_tokens=512)
        if output == '':
            continue
        response = output.strip()
        item = OrderedDict()
        item['id'] = data['id']
        item['source'] = data['source']
        item['task'] = data['task']
        item['type'] = data['type']
        item['judge'] = None
        item['correct_answer'] = data['answer']
        if data['task'] == 'Legal Case Retrieval':
            item['pred_answer'] = extract_case_answer(response)
        elif data['task'] == 'Legal Article Extraction':
            item['pred_answer'] = extract_law_answer(response)
        elif data['task'] == 'Version Control':
            item['pred_answer'] = extract_version_control_answer(response)
        elif data['source'] == 'Finance' and data['task'] == "Trend Analysis":
            try:
                item['pred_answer'] = extract_qa_answer_trend_analysis(response)[0]
            except:
                item['pred_answer'] = None
        elif data['source'] == 'Finance':
            try:
                item['pred_answer'] = extract_qa_answer(response)[0]
            except:
                item['pred_answer'] = None
        else:
            item['pred_answer'] = multiple_choice_answer(response)

        item['response'] = response

        if data['task'] == "Metric Calculation":
            pred_val = normalize_number(item['pred_answer'])
            correct_val = normalize_number(item['correct_answer'])
            if pred_val is None or correct_val is None:
                item['judge'] = False
            else:
                item['judge'] = (abs(pred_val - correct_val) / abs(correct_val) < 0.05)
        elif data['task'] == "Cross-Company Comparison":
            try:
                pred_val = normalize_number(item['pred_answer'])
                correct_val = normalize_number(item['correct_answer'])
                if pred_val is not None and correct_val is not None:
                    item['judge'] = (abs(pred_val - correct_val) / abs(correct_val) < 0.05)
            except:
                item['judge'] = item['pred_answer'] == item['correct_answer']
        elif data['task'] == 'Version Control':
            correct_answer = ast.literal_eval(item['correct_answer'])
            item['judge'] = compute_jaccard_score(correct_answer, item['pred_answer'])
        else:
            item['judge'] = item['pred_answer'] == item['correct_answer']

        if item['judge'] == None:
            item['judge'] = False
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()


def load_data(source, split='train'):
    if source.endswith('.jsonl'):
        with open(source, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
    else:
        dataset = load_dataset(source, split=split)

    data_all = [{
        "id": item["id"],
        "source": item["source"],
        "task": item["task"],
        "type": item["type"],
        "instruction": item["instruction"],
        "context": item["context"],
        "question": item["question"],
        "options": "\n".join(item["options"]),
        "answer": item["answer"]
    } for item in dataset]

    return data_all



def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    if args.with_context == 0:
        out_file = os.path.join(args.save_dir, f"{args.model}_no_context.jsonl")
    else:
        out_file = os.path.join(args.save_dir, f"{args.model}.jsonl")

    data_list = load_data(args.data_dir)

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_list:
        if item["id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results_rag")
    parser.add_argument("--model", "-m", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--with_context", "-c", type=bool, default=True)
    parser.add_argument("--n_proc", "-n", type=int, default=4)
    parser.add_argument("--data_dir", "-d", type=str, default="datasets/dataset/splits/test_sample.jsonl")
    args = parser.parse_args()
    main()