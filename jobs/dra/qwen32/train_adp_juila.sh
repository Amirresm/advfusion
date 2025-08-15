#!/usr/bin/env bash

#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=O-%x.%j.out
#SBATCH --error=O-%x.%j.err

PROJECT_ROOT="/home/amirresm/files/advfusion"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/jobs/dra/_setup.sh"

echo "Starting job on '$MACHINE' at $(date) in project root: $PROJECT_ROOT"

lang="julia"

OUTPUT_DIR="/scratch/amirresm/outputs/advfusion/train_adp_qwen32_ct_${lang}"
mkdir -p "$OUTPUT_DIR"
rm "$OUTPUT_DIR"/job.log || true
exec > >(tee -a "$OUTPUT_DIR/job.log") 2>&1
pip freeze >"$OUTPUT_DIR/requirements.txt"

model_path="$STORAGE_ROOT/models/Qwen/Qwen2.5-Coder-32B"
ds_path="$STORAGE_ROOT/data/ct_dataset/${lang}"

nvidia-smi

python -m scripts.train \
	--model_name_or_path "${model_path}" \
	--q "4bit" \
	--lib "adp" \
	--peft "seq_bn" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--train_text_max_length 1024 \
	--train_target_max_length 1024 \
	--do_train \
	--train_completions_only False \
	--train_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--do_eval \
	--eval_batch_size 4 \
	--logging_steps 0.05 \
	--eval_steps 0.1 \
	--valid_text_max_length 1024 \
	--valid_target_max_length 1024 \
	--gen_batch_size 4 \
	--test_text_max_length 2048 \
	--test_target_max_length 2048 \
	--output_dir "${OUTPUT_DIR}"
