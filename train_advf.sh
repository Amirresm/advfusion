#! /usr/bin/env bash

target_adapter_path="results/output_adp_go"

adapter_path_list="\
results/output_adp_javascript,\
results/output_adp_python,\
${target_adapter_path},\
"

python -m scripts.train_advf /home/amirreza/projects/ai/models/llm/llama-3.2-3B \
	--q 4bit \
	--adapter_path_list "$adapter_path_list" \
	--target_adapter_path "$target_adapter_path" \
	--dataset_name_or_path /home/amirreza/projects/ai/data/CodeSearchNet/go \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 500 \
	--max_eval_samples 0 \
	--max_test_samples 2 \
	--chunk_size 256 \
	--preload_advf_from "results/output_advf" \
	--output_dir "results/output_advf"
# --do_train \
