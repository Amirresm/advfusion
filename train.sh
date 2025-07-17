#! /usr/bin/env bash

python -m scripts.train /home/amirreza/projects/ai/models/llm/llama-3.2-3B \
	--q 4bit \
	--lib adp \
	--peft lora \
	--dataset_name_or_path /home/amirreza/projects/ai/data/CodeSearchNet/python \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--input_column code \
	--target_column docstring \
	--max_train_samples 1000 \
	--max_eval_samples 50 \
	--max_test_samples 20 \
	--chunk_size 256 \
	--do_train \
	--output_dir output_adp
# --preload_peft_from output_adp \
