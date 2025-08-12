#!/usr/bin/env bash

#SBATCH --time=20:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1

PROJECT_ROOT="/home/amirresm/files/advfusion"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/jobs/dra/_setup.sh"

echo "Starting job on '$MACHINE' at $(date) in project root: $PROJECT_ROOT"

OUTPUT_DIR="/scratch/amirresm/outputs/advfusion/train_adp_qwen32_csn_python"
mkdir -p "$OUTPUT_DIR"
rm "$OUTPUT_DIR"/job.log || true
exec > >(tee -a "$OUTPUT_DIR/job.log") 2>&1
pip freeze >"$OUTPUT_DIR/requirements.txt"

lang="python"
model_path="$STORAGE_ROOT/models/Qwen/Qwen2.5-Coder-32B"
ds_path="$STORAGE_ROOT/data/CodeSearchNet/${lang}"

nvidia-smi

python -m scripts.train \
	--model_name_or_path "${model_path}" \
	--q "4bit" \
	--lib "adp" \
	--peft "seq_bn_inv" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--chunk_size 512 \
	--do_train \
	--max_train_samples 2000 \
	--max_eval_samples 20 \
	--max_test_samples 40 \
	--train_completions_only \
	--train_batch_size 4 \
	--do_eval \
	--gen_pre_train_max_samples 10 \
	--eval_batch_size 4 \
	--output_dir "${OUTPUT_DIR}"
