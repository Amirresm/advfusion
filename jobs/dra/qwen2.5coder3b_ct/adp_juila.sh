#!/usr/bin/env bash

#SBATCH --time=5:00:00
#SBATCH --account=rrg-fard
#SBATCH --mem-per-cpu=16000M
#SBATCH --gpus-per-node=1
#SBATCH --output=O-%x.%j.out

echo $SLURM_TMPDIR
pwd
ls
ls ..
ls ../..

PROJECT_ROOT="/home/amirresm/files/advfusion"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/jobs/dra/_setup.sh"

echo "Starting job on '$MACHINE' at $(date) in project root: $PROJECT_ROOT"

lang="julia"

OUTPUT_DIR="/scratch/amirresm/outputs/advfusion/qwen2.5coder3b_ct/adp_${lang}"
mkdir -p "$OUTPUT_DIR"
rm "$OUTPUT_DIR"/job.log || true
exec > >(tee -a "$OUTPUT_DIR/job.log") 2>&1
pip freeze >"$OUTPUT_DIR/requirements.txt"

model_path="$STORAGE_ROOT/models/Qwen/Qwen2.5-Coder-3B"
ds_path="$STORAGE_ROOT/data/ct_dataset/${lang}"

python -m scripts.train \
	--model_name_or_path "${model_path}" \
	--q "4bit" \
	--lib "adp" \
	--peft "seq_bn" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--train_text_max_length 4096 \
	--train_target_max_length 4096 \
	--train_max_length 8192 \
	--do_train \
	--train_completions_only False \
	--train_batch_size 4 \
	--gradient_accumulation_steps 1 \
	--do_eval \
	--eval_batch_size 4 \
	--logging_steps 0.05 \
	--eval_steps 0.2 \
	--valid_text_max_length 2048 \
	--valid_target_max_length 2048 \
	--gen_batch_size 8 \
	--test_text_max_length 4096 \
	--test_target_max_length 2048 \
	--output_dir "${OUTPUT_DIR}"
