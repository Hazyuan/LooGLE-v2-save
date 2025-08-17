import os
import argparse
import subprocess

def run_eval_on_jsonl_files(folder_path, eval_script_path):
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                input_path = os.path.join(dirpath, filename)
                print(f"Running eval on: {input_path}")
                subprocess.run(["python", eval_script_path, "--input_path", input_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run eval.py on all .jsonl files in a folder")
    parser.add_argument("--folder", required=True, help="Path to the folder containing .jsonl files")
    parser.add_argument("--eval_script", default="/home/ubuntu/Web/hzy/LooGLE-v2/eval/eval.py", help="Path to the eval.py script")

    args = parser.parse_args()
    run_eval_on_jsonl_files(args.folder, args.eval_script)
