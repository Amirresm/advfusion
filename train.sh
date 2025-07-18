#! /usr/bin/env bash

python -m scripts.train /home/amirreza/projects/ai/models/llm/llama-3.2-3B \
	--q 4bit \
	--lib adp \
	--peft lora \
	--dataset_name_or_path /home/amirreza/projects/ai/data/CodeSearchNet/javascript \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 2000 \
	--max_eval_samples 100 \
	--max_test_samples 20 \
	--chunk_size 256 \
	--do_train \
	--output_dir "results/output_adp_javascript"
	# --preload_peft_from "results/output_adp" \
