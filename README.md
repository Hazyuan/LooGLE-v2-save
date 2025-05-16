# LooGLE-v2  
**LooGLE v2: A novel real-world benchmark for long-dependency understanding**

## Evaluation

First, create a conda environment and install the required dependencies:

```bash
conda create -n loogle python=3.10
conda activate loogle
pip install vllm
```

Then, clone the benchmark repository:

```bash
git clone https://github.com/GraphPKU/LooGLE-v2.git
cd LooGLE-v2
```

### Download the Dataset

You can download the benchmark dataset into the `./datasets` directory with the following command:

```bash
git clone https://huggingface.co/datasets/GraphPKU/LooGLE-v2 ./datasets/LooGLE-v2
```

### Example: Evaluation with Llama-3.1-8B-Instruct

We take `Llama-3.1-8B-Instruct` as an example for inference.  
First, launch the model server using `vllm serve`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --api-key GraphPKU \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max_model_len 131072 \
  --trust-remote-code
```

> Note: `--tensor-parallel-size` should be set to the number of available GPUs.

### Prediction

To run predictions on the benchmark using your model:

```bash
python predict.py \
  --model Llama-3.1-8B-Instruct \
  --data_dir ./datasets/LooGLE-v2
```

### Evaluation

After inference is complete, run the evaluation script:

```bash
python eval/eval.py \
  --input_path ./results/Llama-3.1-8B-Instruct.jsonl
```

This will compute accuracy and other metrics for the model's performance on LooGLE-v2.
