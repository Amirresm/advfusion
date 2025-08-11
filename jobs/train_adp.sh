#!/usr/bin/env bash

#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1

MACHINE_NAME=$(uname -n)
IS_LOCAL=false
if [[ "$MACHINE_NAME" == "voyager" ]]; then
	IS_LOCAL=true
fi

if [[ "$IS_LOCAL" == true ]]; then
	PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
	STORAGE_ROOT="/mnt/storage"
	OUTPUT_DIR="$PROJECT_ROOT/results/jobs/train_adp"
	mkdir -p "$OUTPUT_DIR"

	lang="julia"
	model_path="$STORAGE_ROOT/ai/models/llm/deepseek-coder-1.3b-base"
	ds_path="$STORAGE_ROOT/ai/data/ct_dataset/${lang}"
else
	# PROJECT_ROOT="/home/amirresm/files/research/advfusion"
	PROJECT_ROOT="/home/amirresm/files/advfusion"
	STORAGE_ROOT="/home/amirresm/projects/def-fard/amirresm"
	OUTPUT_DIR="/scratch/amirresm/outputs/advfusion/train_adp"
	mkdir -p "$OUTPUT_DIR"

	lang="python"
	model_path="$STORAGE_ROOT/models/deepseek-ai/deepseek-coder-7b-base-v1.5"
	ds_path="$STORAGE_ROOT/data/CodeSearchNet/${lang}"

	rm "$OUTPUT_DIR"/job.log || true
	exec > >(tee -a "$OUTPUT_DIR/job.log") 2>&1

	echo "Loading modules..."
	module load StdEnv/2023 gcc cuda arrow python/3.13 scipy-stack
	ENV_PATH=$SLURM_TMPDIR/env
	if [ -f $ENV_PATH ]; then
		echo "Activating existing virtual environment in '$ENV_PATH'..."
		source "$ENV_PATH/bin/activate"
	else
		echo "Creating virtual environment in '$ENV_PATH'..."
		virtualenv --no-download "$ENV_PATH"
		source "$ENV_PATH/bin/activate"
	fi
	cd "$PROJECT_ROOT"
	pwd

	echo "Upgrading pip..."
	pip install --no-index --upgrade pip
	echo "Installing dependencies..."
	pip install --no-index -r "$PROJECT_ROOT/requirements.txt"
	pip install "$PROJECT_ROOT/adapters-1.2.0-py3-none-any.whl"

	pip freeze >"$OUTPUT_DIR/requirements.txt"
fi

echo "Starting job on '$MACHINE_NAME' at $(date) in project root: $PROJECT_ROOT"

python -m scripts.train \
	--model_name_or_path "${model_path}" \
	--q "4bit" \
	--lib "adp" \
	--peft "seq_bn_inv" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 1000 \
	--max_eval_samples 20 \
	--max_test_samples 20 \
	--chunk_size 2048 \
	--do_train \
	--train_completions_only \
	--train_batch_size 2 \
	--do_eval \
	--eval_batch_size 4 \
	--gen_pre_train_max_samples 0 \
	--output_dir "${OUTPUT_DIR}"
