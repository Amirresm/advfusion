from collections import defaultdict
import json
import os

import torch
from transformers.modeling_utils import PreTrainedModel
from rich import print as rprint

from src.dataset.constants import TARGET_COLUMN, TEXT_COLUMN

from src.evaluate.evaluate import calc_all_metrics
from src.utils.jsonl import write_jsonl


def print_generation_result(row):
    index = row["index"]
    input = row["input"]
    target = row["target"]
    token_count = row["token_count"]
    generation = row["generation"]
    metrics = row["metrics"]
    rprint(f"[red]{'-' * 20}Sample {index + 1}:[/red]")
    rprint("[cyan]-------Input text:[/cyan]")
    print(input)
    rprint("[cyan]-------Target text:[/cyan]")
    print(target)

    rprint(f"[green]-------Generated text ({token_count} tokens):[/green]")
    print(generation)

    rprint("[green]-------Metrics:[/green]")
    for k, v in metrics.items():
        try:
            rprint(f"{k}: {float(v):.5f}")
        except:
            rprint(f"{k}: {v}")


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
    model: PreTrainedModel,
    tokenizer,
    raw_dataset,
    num_samples,
    batch_size,
    max_new_tokens,
    save_path=None,
):
    if "test" not in raw_dataset or num_samples is not None and num_samples <= 0:
        return
    samples = (
        raw_dataset["test"][:num_samples]
        if num_samples
        else raw_dataset["test"]
    )

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

    generations_path = f"{save_path}_samples.jsonl" if save_path else None
    if generations_path and os.path.exists(generations_path):
        os.remove(generations_path)

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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
        )
        prompt_length = len(tokenized["input_ids"][0])
        for i, gen in enumerate(generations):
            index = j * batch_size + i
            input = batch[i][0]
            target = batch[i][1]

            new_tokens = gen
            new_tokens = new_tokens[prompt_length:]
            generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
            token_count = len(new_tokens)

            metrics = calc_all_metrics([generation], [target])

            result_row = {
                "index": index,
                "input": input,
                "target": target,
                "generation": generation,
                "token_count": token_count,
                "metrics": metrics,
            }
            results.append(result_row)
            result_row = result_row.copy()

            pretty_metrics = {}
            for k, v in metrics.items():
                if type(v) is float:
                    pretty_metrics[k] = f"{v:.5f}"
                if type(v) is list:
                    pretty_metrics[k] = [
                        f"{x:.5f}" if type(x) is float else x for x in v
                    ]
                else:
                    pretty_metrics[k] = str(v)
            result_row["metrics"] = pretty_metrics
            print_generation_result(result_row)
            if generations_path is not None:
                write_jsonl(generations_path, [result_row], append=True)

    if len(results) == 0:
        return

    total_metrics = calc_all_metrics(
        [r["generation"] for r in results],
        [r["target"] for r in results],
    )

    averages = defaultdict(float)
    for res in results:
        for k, v in res["metrics"].items():
            if type(v) is float:
                averages[k] += v
    for k in averages:
        averages[k] /= len(results)
    averages = dict(averages)

    rprint(f"\n[red]{'Metrics Summary':-^40}[/red]")
    rprint("Total metrics:", total_metrics)
    rprint("Averages:", averages)

    metrics = {
        "total_metrics": total_metrics,
        "averages": averages,
        "metadata": {
            "batch_size": batch_size,
            "num_samples": len(results),
        },
    }

    if save_path is not None:
        file_path = f"{save_path}_metrics.json"
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)

    tokenizer.padding_side = old_pad_side
