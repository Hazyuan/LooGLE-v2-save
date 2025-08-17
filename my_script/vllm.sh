nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 128000 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  > server_distill-Qwen32.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8056 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 129000 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  > log\server_llama3-8B.log 2>&1 &

nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 128000 \
  --dtype float16 \
  --tensor-parallel-size 8 \
  > log\server_llama3-70B.log 2>&1 &

nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 128000 \
  --dtype float16 \
  --tensor-parallel-size 8 \
  > log\server_qwen72B.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --port 8056 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32000 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  > server_dp-v2-lite.log 2>&1 &

# 部分模型
--trust-remote-code
  
  python pred/pred.py --model DeepSeek-V2-Lite-Chat --data_dir ./datasets/LooGLE-v2

nohup python pred/pred_cot.py --model Llama-3.1-8B-Instruct --data_dir ./datasets/LooGLE-v2 --save_dir results_cot --use_cot > log/pred_cot_llama3-8B.log 2>&1 &

nohup python pred/pred.py --model Llama-3.1-8B-Instruct --data_dir ./datasets/code_merged.jsonl --save_dir results_code > log/pred_code_llama3-8B.log 2>&1 &
nohup python pred/pred.py --model Llama-3.3-70B-Instruct --data_dir ./datasets/code_merged.jsonl --save_dir results_code > log/pred_code_llama3-70B.log 2>&1 &

nohup python pred/pred.py --model Qwen2.5-72B-Instruct --data_dir ./datasets/code_merged.jsonl --save_dir results_code > log/pred_code_qwen72B.log 2>&1 &

nohup python pred/pred.py --model glm-4-9b-chat-1m --data_dir ./datasets/LooGLE-v2 --save_dir results --use_cot > log/glm9b-1m.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server   --model Qwen/Qwen2.5-32B-Instruct   --port 8056   --host 0.0.0.0   --gpu-memory-utilization 0.9   --max-model-len 130000   --dtype float16   --tensor-parallel-size 4   > log/server_qwen2-32b.log 2>&1 &
nohup python pred/pred.py --model Qwen2.5-72B-Instruct --data_dir ./datasets/LooGLE-v2 --save_dir results > log/pred_qwen_72b.log 2>&1 &


nohup python pred/pred.py --model Llama-3.1-70B-Instruct --data_dir ./datasets/code_merged.jsonl --save_dir results_code > log/pred_code_llama3-70B.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server   --model NousResearch/Yarn-Mistral-7b-128k   --port 8000   --host 0.0.0.0   --gpu-memory-utilization 0.9   --max-model-len 128000   --dtype float16   --tensor-parallel-size 4   > log/server_yarn7b.log 2>&1 &

Qwen2.5-7B-Instruct-1M

nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server   --model THUDM/glm-4-9b-chat   --port 8000   --host 0.0.0.0   --gpu-memory-utilization 0.9   --max-model-len 128000   --dtype float16   --tensor-parallel-size 4   > log/server_glm-9b.log 2>&1 &

nohup python pred/pred_cot.py --model GLM-4-9b-chat --data_dir ./datasets/LooGLE-v2 --save_dir results_cot --use_cot > log/pred_cot_glm9b.log 2>&1 &

nohup python pred/pred.py --model GLM-4-9b-chat --data_dir ./datasets/LooGLE-v2 --save_dir results_32k > log/pred_32k_glm9b.log 2>&1 &

nohup python pred/pred_cot.py --model Qwen2.5-32B-Instruct --data_dir ./datasets/LooGLE-v2 --save_dir results_cot --use_cot > log/pred_cot_qwen32b.log 2>&1 &

nohup python pred/pred_cot.py --model Yarn-Mistral-7b-128k --data_dir ./datasets/LooGLE-v2 --save_dir results_cot --use_cot > log/pred_cot_yarn.log 2>&1 &

nohup python pred/pred.py --model Yarn-Mistral-7b-128k --data_dir ./datasets/LooGLE-v2 --save_dir results_128k --use_cot > log/pred_128k_yarn.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server   --model Qwen/Qwen2.5-7B-Instruct-1M   --port 8056   --host 0.0.0.0   --gpu-memory-utilization 0.9   --max-model-len 1000000   --dtype float16   --tensor-parallel-size 4   > log/server_qwen7b-1m.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server   --model THUDM/glm-4-9b-chat-1m   --port 8000   --host 0.0.0.0   --gpu-memory-utilization 0.9   --max-model-len 1000000   --dtype float16   --tensor-parallel-size 4   > log/server_glm9b-1m.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server   --model mistralai/Mistral-7B-Instruct-v0.2   --port 8056   --host 0.0.0.0   --gpu-memory-utilization 0.9   --max-model-len 32000   --dtype float16   --tensor-parallel-size 4   > log/server_mistral7b-1m.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server   --model microsoft/Phi-3-medium-128k-instruct   --port 8056   --host 0.0.0.0   --gpu-memory-utilization 0.9   --max-model-len 128000   --dtype float16   --tensor-parallel-size 2   > log/server_phi3-1m.log 2>&1 &
nohup python -m vllm.entrypoints.openai.api_server   --model microsoft/Phi-3-medium-128k-instruct   --port 8056   --host 0.0.0.0   --gpu-memory-utilization 0.9   --max-model-len 128000   --dtype float16   --tensor-parallel-size 2 --pipeline-parallel-size 4  > log/server_phi3-1m.log 2>&1 &

nohup python pred/pred.py --model Mistral-7B-Instruct-v0.2 --data_dir ./datasets/LooGLE-v2 --save_dir results_32k > log/pred_32k_mistral_7b.log 2>&1 &

nohup python pred/pred.py --model Phi-3-medium-128k-instruct --data_dir ./datasets/LooGLE-v2 --save_dir results_128k > log/pred_128k_phi.log 2>&1 &

nohup python pred/pred_evi.py --model GLM-4-9b-chat --data_dir ./datasets/LooGLE-v2 --save_dir results_evi2 --only_evidence > log/pred_evi_glm9b.log 2>&1 &

