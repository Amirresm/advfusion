#!/usr/bin/env bash

#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1

source "$(dirname "$0")/_setup.sh"

echo "Starting job on '$MACHINE' at $(date) in project root: $PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/results/jobs/train_qwen32"
mkdir -p "$OUTPUT_DIR"

lang="julia"
# model_path="$STORAGE_ROOT/ai/models/llm/deepseek-coder-1.3b-base"
model_path="$STORAGE_ROOT/ai/models/llm/Qwen2.5-Coder-1.5B"
# model_path="$STORAGE_ROOT/ai/models/llm/CodeLlama-7b"
# ds_path="$STORAGE_ROOT/ai/data/CodeSearchNet/${lang}"
ds_path="$STORAGE_ROOT/ai/data/ct_dataset/${lang}"

benchmark_dataset_name_or_path="/home/amirreza/projects/ubc/multipl_e_ct/data/processed/ct_bench_dataset_go_julia.jsonl"

python -m scripts.train \
	--model_name_or_path "${model_path}" \
	--q "4bit" \
	--lib "adp" \
	--peft "seq_bn" \
	--preload_peft_from "$OUTPUT_DIR" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 1000 \
	--max_eval_samples 20 \
	--max_test_samples 0 \
	--epochs 3 \
	--chunk_size 0 \
	--train_text_max_length 1024 \
	--train_target_max_length 1024 \
	--train_max_length 2048 \
	--train_completions_only False \
	--do_train 0 \
	--debug \
	--learning_rate 1e-4 \
	--train_batch_size 2 \
	--do_eval 0 \
	--logging_steps 0.1 \
	--eval_steps 0.95 \
	--eval_batch_size 2 \
	--gen_pre_train_max_samples 0 \
	--gen_batch_size 16 \
	--benchmark_dataset_name_or_path "${benchmark_dataset_name_or_path}" \
	--benchmark_dataset_type "CodeTranslationBench" \
	--benchmark_do_sample 1 \
	--benchmark_temperature 0.8 \
	--benchmark_top_p 0.95 \
	--benchmark_top_k 0 \
	--benchmark_n_per_sample 10 \
	--output_dir "${OUTPUT_DIR}"
