#! /usr/bin/env bash

target_adapter_path="results/qwen/python"

adapter_path_list="\
results/qwen/go,\
results/qwen/java,\
results/qwen/php,\
${target_adapter_path},\
"

python -m scripts.train_advf /home/amirreza/projects/ai/models/llm/Qwen2.5-Coder-1.5B \
	--q 4bit \
	--adapter_path_list "$adapter_path_list" \
	--target_adapter_path "$target_adapter_path" \
	--dataset_name_or_path /home/amirreza/projects/ai/data/CodeSearchNet/python \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 1000 \
	--max_eval_samples 10 \
	--max_test_samples 20 \
	--chunk_size 256 \
	--do_train \
	--output_dir "results/qwen/advf"
# --preload_advf_from "results/output_advf" \
