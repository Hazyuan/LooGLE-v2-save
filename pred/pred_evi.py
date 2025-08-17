import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
from collections import OrderedDict
from datetime import datetime
import requests
from metrics import *
from utils import *
import ast


def query_llm(prompt, data, model, client, temperature=0.1, max_new_tokens=32, stop=None):
    # truncate

    #     # Debug: log the last 1000 chars of context

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

def judge_answer(data, item):
    task = data['task']
    pred = item['pred_answer']
    truth = item['correct_answer']

    if task in {"Metric Calculation", "Cross-Company Comparison"}:
        pred_val = normalize_number(pred)
        correct_val = normalize_number(truth)
        if pred_val is None or correct_val is None:
            return False
        if correct_val == 0:
            return pred_val == 0
        return abs(pred_val - correct_val) / abs(correct_val) < 0.05

    elif task == "Version Control":
        try:
            correct_list = ast.literal_eval(truth)
            return compute_jaccard_score(correct_list, pred)
        except:
            return False

    else:
        return pred == truth


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
        prompt , remove_part= format_prompt(model, data, tokenizer, max_len, args.with_context,args.use_cot)
        # DEBUG
        # with open("context_debug.log", "a", encoding="utf-8") as logf:
        #     logf.write(f"{data.get('id', 'unknown_id')}\t{prompt[-1000:]}\n")

        if args.only_evidence:
            if data["source"] != "Finance":
                #rint(f"Skipping item {data['id']} as it is not from Finance source.")
                continue
            prompt_only_evi = data["instruction"].format(context="""I will directly provide the relevant data needed to answer the question. The data will not be labeled or explained, so please rely on the question to determine what to use. When answering with specific numerical values, always convert large numbers to millions (M) correctly:
- 1M = 1,000,000.0
- For example, 2,420,000,000.0 = 2420M, not 2.42M
- Always divide by 1,000,000 when converting to M
- Use integer values for Million""", options=data["options"], question=data["question"])
            prompt_only_evi += "\n\n<information>" + data["evidence"]+ "\n</information>\n"
            output = query_llm(prompt_only_evi, data, model, client, temperature=args.temperature, max_new_tokens=2048)

        elif args.use_cot:
            output = query_llm(prompt, data, model, client, temperature=args.temperature, max_new_tokens=4096)
        else:
            output = query_llm(prompt, data, model, client, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        if output == '':
            continue
        # DEBUG
        with open("output.log", "a", encoding="utf-8") as f_log:
            f_log.write(f"ID: {data.get('id', 'unknown')}\n")
            f_log.write(f"Output:\n{output}\n")
            f_log.write("="*80 + "\n")
        
        if args.use_cot:
            response = output.strip()
            prompt_extract = data["instruction"].format(context="The text is too long and omitted here.", options=data["options"], question=data["question"])
            lower_prompt = prompt_extract.lower()
            idx = lower_prompt.rfind("please")
            if idx != -1:
                prompt_extract = prompt_extract[:idx].rstrip()
            prompt_extract += "\n\nLet's think step by step."
            prompt_extract += "\n\nAnalysis: "
            prompt_extract += response
            prompt_extract += "\n\nNow, based on the analysis," + remove_part
            # DEBUG
            with open("context_debug.log", "a", encoding="utf-8") as logf:
                logf.write(f"COT {data.get('id', 'unknown_id')}\t{prompt_extract[-2000:]}\n")
            output = query_llm(prompt_extract, data, model, client, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
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

        task = data['task']
        if task in extractor_map:
            try:
                item['pred_answer'] = extractor_map[task](response)
            except Exception:
                item['pred_answer'] = None
        else:
            item['pred_answer'] = multiple_choice_answer(response)

        item['response'] = response
        item['judge'] = judge_answer(data, item)
        if item['judge'] is None:
            item['judge'] = False

        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()


def load_data(source, split='test'):
    if source.endswith('.jsonl'):
        with open(source, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
    else:
        dataset = load_dataset(source, split=split)
    data_all = []
    for item in dataset:
        options = item['options']
        if isinstance(options,list):
            options = "\n".join(options)
        data_all.append({
            "id": item["id"],
            "source": item["source"],
            "task": item["task"],
            "type": item["type"],
            "instruction": item["instruction"],
            "context": item["context"],
            "question": item["question"],
            "evidence": item.get("evidence", ""),
            "options": options,
            "answer": item["answer"]
        })

    return data_all



def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    if args.with_context == 0:
        out_file = os.path.join(args.save_dir, f"{args.model}_no_context.jsonl")
    else:
        out_file = os.path.join(args.save_dir, f"{args.model}.jsonl")

    data_list = load_data(args.data_dir)

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

    parser.add_argument(
        "--save_dir", "-s",
        type=str,
        default="results",
        help="Directory to save the prediction results."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Llama-3.1-8B-Instruct",
        help="Name or ID of the model to use. Must match entry in config/models.jsonl."
    )
    parser.add_argument(
        "--with_context", "-c",
        type=bool,
        default=True,
        help="Whether to include context in the prompt. Use False to test model without providing external context."
    )
    parser.add_argument(
        "--n_proc", "-n",
        type=int,
        default=4,
        help="Number of parallel processes to use for evaluation."
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="datasets/dataset/splits/test_sample.jsonl",
        help="Path to the input dataset. Can be a .jsonl file or HuggingFace dataset name."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for each model output."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation. Higher values increase randomness."
    )
    parser.add_argument(
        "--use_cot",
        action="store_true",
        help="Enable Chain-of-Thought (CoT) prompting. If set, will modify prompt to encourage step-by-step reasoning."
    )
    parser.add_argument(
        "--only_evidence",
        action="store_true",
        help="Test a model without context, directly provide the context."
    )
    args = parser.parse_args()
    main()
