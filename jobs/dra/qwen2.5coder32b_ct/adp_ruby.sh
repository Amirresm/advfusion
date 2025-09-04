#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --account=rrg-fard
#SBATCH --mem-per-cpu=16000M
#SBATCH --gpus-per-node=1
#SBATCH --output=O-%x.%j.out

echo "TMP DIR: ${SLURM_TMPDIR}"
echo "PWD: ${PWD}"
echo "HOST: $(hostname)"
echo "ls, ls .., ls ../.."
ls
ls ..
ls ../..
echo "End of debug info"

PROJECT_ROOT="/home/amirresm/files/advfusion"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/jobs/dra/_setup.sh"

echo "Starting job on '$MACHINE' at $(date) in project root: $PROJECT_ROOT"

lang="ruby"

OUTPUT_DIR="/scratch/amirresm/outputs/advfusion/qwen2.5coder32b_ct/adp_${lang}"
mkdir -p "$OUTPUT_DIR"
rm "$OUTPUT_DIR"/job.log || true
exec > >(tee -a "$OUTPUT_DIR/job.log") 2>&1
pip freeze >"$OUTPUT_DIR/requirements.txt"

model_path="$STORAGE_ROOT/models/Qwen/Qwen2.5-Coder-32B"
ds_path="$STORAGE_ROOT/data/ct_dataset/${lang}"

benchmark_dataset_name_or_path="$STORAGE_ROOT/data/ct_bench_dataset/ct_bench_dataset_all_${lang}.jsonl"

python -m scripts.train \
	--model_name_or_path "${model_path}" \
	--q "4bit" \
	--lib "adp" \
	--peft "seq_bn" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 9999999 \
	--max_eval_samples 9999999 \
	--max_test_samples 9999999 \
	--chunk_size 0 \
	--train_text_max_length 4096 \
	--train_target_max_length 4096 \
	--train_max_length 8192 \
	--epochs 2 \
	--do_train \
	--train_completions_only False \
	--train_batch_size 1 \
	--gradient_accumulation_steps 4 \
	--learning_rate 1e-4 \
	--do_eval \
	--eval_batch_size 1 \
	--logging_steps 0.05 \
	--eval_steps 0.2 \
	--valid_text_max_length 2048 \
	--valid_target_max_length 2048 \
	--gen_pre_train_max_samples 32 \
	--gen_batch_size 16 \
	--test_text_max_length 4096 \
	--test_target_max_length 2048 \
	--benchmark_dataset_name_or_path "${benchmark_dataset_name_or_path}" \
	--benchmark_do_sample 1 \
	--benchmark_temperature 0.3 \
	--benchmark_top_p 0.9 \
	--benchmark_top_k 50 \
	--benchmark_n_per_sample 10 \
	--output_dir "${OUTPUT_DIR}"
