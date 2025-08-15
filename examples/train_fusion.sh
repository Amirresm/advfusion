#! /usr/bin/env bash

adapter_path_list=(
	"results/deepseek-coder/rust"
	"results/deepseek-coder/ruby"
	"results/deepseek-coder/scala"
	"results/deepseek-coder/lua"
)

model_path="/mnt/storage/ai/models/llm/deepseek-coder-1.3b-base"
ds_path="/mnt/storage/ai/data/ct_dataset/lua"
output_dir="results/deepseek-coder_fusion/lua"

echo "Model Path: $model_path"
python -m scripts.train_fusion \
	--model_name_or_path "$model_path" \
	--q 4bit \
	--adapter_path_list "${adapter_path_list[@]}" \
	--dataset_name_or_path "$ds_path" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 1000 \
	--max_eval_samples 20 \
	--max_test_samples 20 \
	--chunk_size 0 \
	--train_text_max_length 512 \
	--train_target_max_length 512 \
	--nodo_train \
	--preload_fusion_from "$output_dir" \
	--train_completions_only False \
	--train_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--do_eval \
	--eval_batch_size 4 \
	--eval_steps 0.2 \
	--valid_text_max_length 512 \
	--valid_target_max_length 512 \
	--gen_pre_train_max_samples 3 \
	--gen_batch_size 8 \
	--test_text_max_length 1024 \
	--test_target_max_length 1024 \
	--output_dir "$output_dir"
