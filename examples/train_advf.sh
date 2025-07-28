#! /usr/bin/env bash

target_adapter_path="results/deepseek-coder/lua"

adapter_path_list=(
	"results/deepseek-coder/rust"
	"results/deepseek-coder/ruby"
	"results/deepseek-coder/julia"
	"results/deepseek-coder/scala"
	$target_adapter_path
)

model_path="/mnt/storage/ai/models/llm/deepseek-coder-1.3b-base"
ds_path="/mnt/storage/ai/data/ct_dataset/lua"
output_dir="results/deepseek-coder_advf/lua"

echo "Model Path: $model_path"
python -m scripts.train_advf \
	--model_name_or_path "$model_path" \
	--q 4bit \
	--adapter_path_list "${adapter_path_list[@]}" \
	--target_adapter_path "$target_adapter_path" \
	--dataset_name_or_path "$ds_path" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 1000 \
	--max_eval_samples 20 \
	--max_test_samples 20 \
	--chunk_size 2048 \
	--do_train \
	--train_completions_only \
	--train_batch_size 1 \
	--do_eval \
	--eval_batch_size 4 \
	--gen_pre_train_max_samples 20 \
	--output_dir "$output_dir"
