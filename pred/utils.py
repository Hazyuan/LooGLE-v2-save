import json
def truncate(model, prompt, tokenizer, max_len):
    use_tiktoken = any(x in model for x in ["gpt", "o1", "o3", "o4", "Fin-R1"])
    if use_tiktoken:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
    else:
        input_ids = tokenizer.encode(prompt)

    if len(input_ids) > max_len:
        half = max_len // 2
        input_ids = input_ids[:half] + input_ids[-half:]
    if use_tiktoken:
        prompt = tokenizer.decode(input_ids)
    else:
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

    return prompt



def format_prompt(model, data, tokenizer, max_len, with_context, use_cot=False):
    if with_context == False:
        data["context"] = "I would not provide you with the context. Please choose the most likely option based on your knowledge and intuition."
    prompt = data["instruction"].format(context=data["context"], options=data["options"], question=data["question"])
    if use_cot:
        lower_prompt = prompt.lower()
        idx = lower_prompt.rfind("please")
        removed_part = ""
        if idx != -1:
            removed_part = prompt[idx:].lstrip()
            prompt = prompt[:idx].rstrip()
        prompt += "\n\nLet's think step by step."
        prompt += "\n\nAnalysis: "

    prompt = truncate(model, prompt, tokenizer, max_len)
    return prompt, removed_part if use_cot else None




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