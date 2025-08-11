#!/usr/bin/env bash

#SBATCH --time=20:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1

PROJECT_ROOT="/home/amirresm/files/advfusion"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/jobs/dra/_setup.sh"

echo "Starting job on '$MACHINE' at $(date) in project root: $PROJECT_ROOT"

lang="python"
model_path="$STORAGE_ROOT/models/deepseek-ai/deepseek-coder-7b-base-v1.5"
ds_path="$STORAGE_ROOT/data/CodeSearchNet/${lang}"

python -m scripts.train \
	--model_name_or_path "${model_path}" \
	--q "4bit" \
	--lib "adp" \
	--peft "seq_bn_inv" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--chunk_size 2048 \
	--do_train \
	--train_completions_only \
	--train_batch_size 2 \
	--do_eval \
	--eval_batch_size 4 \
	--output_dir "${OUTPUT_DIR}"
# --max_train_samples 1000 \
# 	--max_eval_samples 20 \
# 	--max_test_samples 20 \
# --gen_pre_train_max_samples 0 \
