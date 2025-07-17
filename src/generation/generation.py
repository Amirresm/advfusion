def generate_raw_samples(
    model, tokenizer, samples, input_column, target_column, batch_size=1
):
    if model.config.pad_token_id is None:
        print("Setting pad_token_id to eos_token_id")
        model.config.pad_token_id = tokenizer.eos_token_id
    if model.generation_config.pad_token_id is None:
        print("Setting generation_config.pad_token_id to eos_token_id")
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batches = []
    for i in range(0, len(samples[input_column]), batch_size):
        batch = []
        for j in range(batch_size):
            if i + j < len(samples[input_column]):
                input = samples[input_column][i + j]
                target = samples[target_column][i + j]
                # input = input.replace(target, "")
                input = f"{input}\nExplanation:\n"
                batch.append((input, target))
        batches.append(batch)

    for j, batch in enumerate(batches):
        tokenized = tokenizer(
            [b[0] for b in batch],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        )
        generations = model.generate(
            input_ids=tokenized["input_ids"].to(model.device),
            attention_mask=tokenized["attention_mask"].to(model.device),
            max_new_tokens=100,
        )
        prompt_length = len(tokenized["input_ids"][0])
        for i, gen in enumerate(generations):
            print("=" * 50)
            print(f"Sample {j * batch_size + i + 1}:")
            input = batch[i][0]
            target = batch[i][1]
            print("-" * 20, f"Input text:\n{input}")
            print("-" * 20, f"Target text:\n{target}")

            new_tokens = gen[prompt_length:]
            generation = tokenizer.decode(new_tokens, skip_special_tokens=False)
            print(
                "-" * 20,
                f"Generated text ({len(new_tokens)} tokens):\n{generation}\n",
            )

    tokenizer.padding_side = old_pad_side
    return

    for i in range(len(samples["code"])):
        print("=" * 50)
        print(f"Sample {i + 1}:")
        input = samples["code"][i]
        target = samples["docstring"][i]
        input = input.replace(target, "")
        input = f"{input}\nExplanation:\n"
        print("-" * 20, f"Input text:\n{input}")
        print("-" * 20, f"Target text:\n{target}")
        tokenized = tokenizer(
            input,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        prompt_length = len(tokenized["input_ids"][0])
        generations = model.generate(
            input_ids=tokenized["input_ids"].to(model.device),
            attention_mask=tokenized["attention_mask"].to(model.device),
            max_new_tokens=100,
        )
        new_tokens = generations[0][prompt_length:]

        generation = tokenizer.decode(new_tokens, skip_special_tokens=False)
        print(
            "-" * 20,
            f"Generated text ({len(new_tokens)} tokens):\n{generation}\n",
        )
