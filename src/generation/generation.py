import os

import torch

from transformers.modeling_utils import PreTrainedModel
from src.dataset.constants import TARGET_COLUMN, TEXT_COLUMN

from src.evaluate.evaluate import calc_all_metrics


def custom_generation_loop(model, tokenizer, sample_sent):
    tokenized = tokenizer(
        sample_sent,
        return_tensors="pt",
    )
    sample_input = tokenized["input_ids"].to(model.device)
    sample_attn = tokenized["attention_mask"].to(model.device)
    print("Sample sentence:", sample_sent)
    print("Sample input:", sample_input)
    out_tokens = []
    for i in range(10):
        out = model(
            input_ids=sample_input,
            attention_mask=sample_attn,
        )
        logits = out.logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        sample_input = torch.cat(
            [sample_input, next_token.unsqueeze(0)],
            dim=1,
        )
        sample_attn = torch.cat(
            [sample_attn, torch.ones((1, 1), device=sample_input.device)],
            dim=1,
        )
        next_token = next_token.item()
        out_tokens.append(next_token)
        token = tokenizer.decode(
            next_token,
            skip_special_tokens=True,
        )
        print(token, end="")

    print("\nGenerated tokens:", out_tokens)


def generate_raw_samples(
    model: PreTrainedModel, tokenizer, samples, batch_size=1, save_path=None
):
    if len(samples[TEXT_COLUMN]) == 0 or len(samples[TARGET_COLUMN]) == 0:
        print("No samples to generate from.")
        return

    input_column = TEXT_COLUMN
    target_column = TARGET_COLUMN
    if model.config.pad_token_id is None:
        print("Setting pad_token_id to eos_token_id")
        model.config.pad_token_id = tokenizer.eos_token_id
    if (
        model.generation_config is not None
        and model.generation_config.pad_token_id is None
    ):
        print("Setting generation_config.pad_token_id to eos_token_id")
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    results = []

    batches = []
    for i in range(0, len(samples[input_column]), batch_size):
        batch = []
        for j in range(batch_size):
            if i + j < len(samples[input_column]):
                input = samples[input_column][i + j]
                target = samples[target_column][i + j]
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
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
        )
        prompt_length = len(tokenized["input_ids"][0])
        for i, gen in enumerate(generations):
            print("=" * 50)
            print(f"Sample {j * batch_size + i + 1}:")
            input = batch[i][0]
            target = batch[i][1]
            print("-" * 20, f"Input text:\n{input}")
            print("-" * 20, f"Target text:\n{target}")

            new_tokens = gen
            new_tokens = new_tokens[prompt_length:]
            generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(
                "-" * 20,
                f"Generated text ({len(new_tokens)} tokens):\n{generation}\n",
            )

            metrics = calc_all_metrics([generation], [target])
            print("-" * 20, f"Metrics:")
            for k, v in metrics.items():
                try:
                    print(f"{k}: {float(v):.5f}")
                except:
                    print(f"{k}: {v}")

            numeric_metrics = {}
            for k, v in metrics.items():
                try:
                    numeric_metrics[k] = float(v)
                except:
                    pass

            results.append(
                {
                    "input": input,
                    "target": target,
                    "generation": generation,
                    "metrics": numeric_metrics,
                }
            )

    if len(results) == 0:
        return

    total_metrics = calc_all_metrics(
        [r["generation"] for r in results],
        [r["target"] for r in results],
    )

    averages = None
    for res in results:
        if averages is None:
            averages = res["metrics"].copy()
        else:
            for k, v in res["metrics"].items():
                averages[k] += v
    if averages is not None:
        for k in averages:
            averages[k] /= len(results)
        print("-" * 20, f"Averages:")
        averages_str = ""
        for k, v in averages.items():
            try:
                averages += f"{k}: {float(v):.5f}\n"
            except:
                averages_str += f"{k}: {v}\n"
        print(averages_str)
        if save_path is not None:
            file_path = f"{save_path}_averages.txt"
            with open(file_path, "w") as f:
                f.write(averages_str)

    print("-" * 20, f"Total metrics:")
    total_metrics_str = ""
    for k, v in total_metrics.items():
        try:
            total_metrics_str += f"{k}: {float(v):.5f}\n"
        except:
            total_metrics_str += f"{k}: {v}\n"
    print(total_metrics_str)
    if save_path is not None:
        file_path = f"{save_path}_total_metrics.txt"
        with open(file_path, "w") as f:
            f.write(total_metrics_str)

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
