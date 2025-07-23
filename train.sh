#! /usr/bin/env bash

python -m scripts.train /home/amirreza/projects/ai/models/llm/llama-3.2-3B \
	--q 4bit \
	--lib "peft" \
	--peft "lora" \
	--dataset_name_or_path /home/amirreza/projects/ai/data/CodeSearchNet/javascript \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 500 \
	--max_eval_samples 0 \
	--max_test_samples 5 \
	--chunk_size 256 \
	--preload_peft_from "results/output_test" \
	--output_dir "results/output_test"
# --do_train \
