from itertools import chain
from src.dataset.constants import TARGET_COLUMN, TEXT_COLUMN

from rich import print


class TokenizeTextPreprocessor:
    def __init__(
        self,
        tokenizer,
        truncate: bool = False,
        text_max_length: int | None = None,
        target_max_length: int | None = None,
        only_completion: bool = False,
    ):
        self.tokenizer = tokenizer
        self.truncate = truncate
        self.text_max_length = text_max_length
        self.target_max_length = target_max_length
        self.only_completion = only_completion

    def get_tokenizer_preprocessor(
        self,
    ):
        tokenizer = self.tokenizer
        truncate = self.truncate
        text_max_length = self.text_max_length
        target_max_length = self.target_max_length
        only_completion = self.only_completion

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
                for toks, attns in zip(
                    target_tok["input_ids"], target_tok["attention_mask"]
                ):
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

                # if target_input_ids[0] == tokenizer.bos_token_id:
                #     target_input_ids = target_input_ids[1:]
                #     target_attention_mask = target_attention_mask[1:]

                input_ids = text_input_ids + target_input_ids
                attention_mask = text_attention_mask + target_attention_mask

                if only_completion:
                    label_prompt = [-100] * len(text_input_ids)
                else:
                    label_prompt = [
                        token if mask == 1 else -100
                        for token, mask in zip(
                            text_input_ids, text_attention_mask
                        )
                    ]
                label_target = [
                    token if mask == 1 else -100
                    for token, mask in zip(
                        target_input_ids, target_attention_mask
                    )
                ]
                new_examples["input_ids"].append(input_ids)
                new_examples["attention_mask"].append(attention_mask)
                new_examples["labels"].append(label_prompt + label_target)

            return new_examples

        return preprocessor


class ChunkTextPreprocessor:
    dropped_chunk_count: int = 0

    def __init__(self, block_size):
        self.block_size = block_size

    def get_group_text_preprocessor(
        self,
    ):
        block_size = self.block_size

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
            total_length = (total_length // block_size) * block_size
            result = {
                k: [
                    t[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
                for k, t in concatenated_examples.items()
            }

            keep_indices = [
                i
                for i, label_block in enumerate(result["labels"])
                if any(label != -100 for label in label_block)
            ]

            num_dropped = len(result["labels"]) - len(keep_indices)
            self.dropped_chunk_count += num_dropped

            result = {
                k: [v[i] for i in keep_indices] for k, v in result.items()
            }

            return result

        return preprocessor

    def report(self):
        if self.dropped_chunk_count > 0:
            print(
                f"Warning: Dropped {self.dropped_chunk_count} chunks ({self.dropped_chunk_count * self.block_size} tokens) due to all labels being -100."
            )
