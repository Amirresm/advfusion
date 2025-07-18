from itertools import chain
from src.dataset.constants import TARGET_COLUMN, TEXT_COLUMN


def get_tokenizer_preprocessor(
    tokenizer,
    truncate: bool = True,
    text_max_length: int | None = 256,
    target_max_length: int | None = 128,
    only_completion: bool = False,
):
    def preprocessor(examples):
        new_examples = {}

        if truncate:
            text_tok = tokenizer(
                examples[TEXT_COLUMN],
                padding="max_length",
                truncation=True,
                max_length=text_max_length,
                padding_side="left",
            )
            target_tok = tokenizer(
                [
                    s + "\n" + tokenizer.eos_token
                    for s in examples[TARGET_COLUMN]
                ],
                padding="max_length",
                truncation=True,
                max_length=target_max_length,
                padding_side="right",
            )
            for toks, attns in zip(target_tok["input_ids"], target_tok["attention_mask"]):
                if attns[-1] == 1 and toks[-1] != tokenizer.eos_token_id:
                    toks[-1] = tokenizer.eos_token_id
        else:
            text_tok = tokenizer(
                examples[TEXT_COLUMN],
                padding=False,
                truncation=False,
            )
            target_tok = tokenizer(
                [
                    s + "\n" + tokenizer.eos_token
                    for s in examples[TARGET_COLUMN]
                ],
                padding=False,
                truncation=False,
            )

        new_examples["input_ids"] = []
        new_examples["attention_mask"] = []
        new_examples["labels"] = []
        for i in range(len(text_tok["input_ids"])):
            text_input_ids = text_tok["input_ids"][i]
            text_attention_mask = text_tok["attention_mask"][i]
            target_input_ids = target_tok["input_ids"][i]
            target_attention_mask = target_tok["attention_mask"][i]
            if target_input_ids[0] == tokenizer.bos_token_id:
                target_input_ids = target_input_ids[1:]
                target_attention_mask = target_attention_mask[1:]

            input_ids = text_input_ids + target_input_ids
            attention_mask = text_attention_mask + target_attention_mask
            if only_completion:
                label_prompt = [-100] * len(text_input_ids)
            else:
                label_prompt = [
                    token if mask == 1 else -100
                    for token, mask in zip(text_input_ids, text_attention_mask)
                ]
            label_target = [
                token if mask == 1 else -100
                for token, mask in zip(target_input_ids, target_attention_mask)
            ]
            new_examples["input_ids"].append(input_ids)
            new_examples["attention_mask"].append(attention_mask)
            new_examples["labels"].append(label_prompt + label_target)

        return new_examples

    return preprocessor


def get_group_text_preprocessor(
    block_size: int,
    # padding,
    # ignore_pad_token_for_loss=False,
):
    def preprocessor(examples):
        keys = [
            k
            for k in [
                "input_ids",
                "attention_mask",
                "labels",
            ]
            if k in examples.keys()
        ]
        concatenated_examples = {k: list(chain(*examples[k])) for k in keys}
        total_length = len(concatenated_examples[keys[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }

        # Filter out blocks with all labels == -100
        keep_indices = [
            i for i, label_block in enumerate(result["labels"])
            if any(label != -100 for label in label_block)
        ]

        # Optionally warn if a large number are dropped
        num_dropped = len(result["labels"]) - len(keep_indices)
        if num_dropped > 0:
            print(f"Warning: {num_dropped} chunks dropped where all labels == -100")

        # Keep only the valid blocks
        result = {
            k: [v[i] for i in keep_indices]
            for k, v in result.items()
        }

        return result

    return preprocessor
