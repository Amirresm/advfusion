#! /usr/bin/env bash

python -m scripts.train /mnt/storage/ai/models/llm/deepseek-coder-1.3b-base \
	--q 4bit \
	--lib "adp" \
	--peft "seq_bn" \
	--dataset_name_or_path /mnt/storage/ai/data/CodeSearchNet/python \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 1000 \
	--max_eval_samples 10 \
	--max_test_samples 20 \
	--chunk_size 256 \
	--preload_peft_from "results/test" \
	--output_dir "results/test"
# --do_train \
# --preload_peft_from "results/qwen/ct_julia" \
# --preload_peft_from "results/output_test" \
#
#
# --dataset_name_or_path /home/amirreza/projects/ubc/ct_dataset/processed_data/julia \
