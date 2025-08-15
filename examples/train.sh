#! /usr/bin/env bash

lang="julia"

model_path="/mnt/storage/ai/models/llm/deepseek-coder-1.3b-base"
ds_path="/mnt/storage/ai/data/ct_dataset/${lang}"
output_dir="results/deepseek-coder/${lang}"

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
	--do_train \
	--notrain_completions_only \
	--train_batch_size 4 \
	--gradient_accumulation_steps 2 \
	--chunk_size 0 \
	--train_text_max_length 512 \
	--train_target_max_length 256 \
	--do_eval \
	--eval_batch_size 64 \
	--eval_steps 100 \
	--gen_pre_train_max_samples 0 \
	--gen_batch_size 64 \
	--output_dir "${output_dir}"
