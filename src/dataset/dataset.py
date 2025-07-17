from itertools import chain
import os
from typing import Callable

from datasets import DatasetDict, load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


def load_raw_dataset(
    dataset_path: str,
    train_file: str | None = None,
    test_file: str | float = 0.2,
    validation_file: str | float = 0.5,
):
    base_path = os.path.expanduser(dataset_path)
    if train_file is None:
        train_file = dataset_path.split("/")[-1]
        assert (
            train_file
        ), "Train file must be specified or derived from dataset_path."
        base_path = os.path.dirname(base_path)

    extention = train_file.split(".")[-1].lower()
    if extention == "jsonl":
        extention = "json"

    splits: list[tuple[str, str | float]] = [
        ("train", train_file),
        ("test", test_file),
        ("validation", validation_file),
    ]

    data_files = {}
    for split, file in splits:
        if file is not None and isinstance(file, str):
            file_path = os.path.join(base_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            data_files[split] = file_path

    raw_dataset = load_dataset("json", data_files=data_files)
    assert type(raw_dataset) is DatasetDict, "Dataset loading failed."

    if "test" not in raw_dataset and isinstance(test_file, float):
        print("Splitting train dataset into train and test sets.")
        raw_dataset = raw_dataset["train"].train_test_split(
            test_size=test_file, seed=42
        )
    if "validation" not in raw_dataset and isinstance(validation_file, float):
        print("Splitting test dataset into test and validation sets.")
        temp_dataset = raw_dataset["test"].train_test_split(
            test_size=validation_file, seed=42
        )
        raw_dataset = DatasetDict(
            {
                "train": raw_dataset["train"],
                "test": temp_dataset["train"],
                "validation": temp_dataset["test"],
            }
        )
    return raw_dataset


type ColumnJoiner = Callable[[str, str], str]
TEXT_COLUMN = "TEXT"
TARGET_COLUMN = "TARGET"
TEXT_TOKENIZED_COLUMN = "input_ids"
TEXT_ATTN_MASK_COLUMN = "attention_mask"
TARGET_TOKENIZED_COLUMN = "TARGET_TOKENIZED"
TARGET_ATTN_MASK_COLUMN = "TARGET_ATTN_MASK"


def get_column_preprocessor(
    text_column: str,
    target_column: str | None = None,
    join_text_target: ColumnJoiner | None = None,
):
    def preprocessor(examples):
        texts, targets = [], []
        for i in range(len(examples[text_column])):
            text = examples[text_column][i]
            target = None

            if target_column is not None:
                if join_text_target:
                    # text = text.replace(examples[target_column][i], "")
                    text = join_text_target(text, examples[target_column][i])
                else:
                    target = examples[target_column][i]

            texts.append(text)
            targets.append(target)

        examples[TEXT_COLUMN] = texts
        if target_column is not None and not join_text_target:
            examples[TARGET_COLUMN] = targets

        return examples

    return preprocessor


def get_tokenizer_preprocessor(
    tokenizer,
    truncate: bool = True,
):
    def preprocessor(examples):
        new_examples = {}
        if truncate:
            text_tok = tokenizer(
                [
                    s + "\n" + tokenizer.eos_token
                    for s in examples[TEXT_COLUMN]
                ],
                padding="longest",  # options: "longest", "max_length", "do_not_pad"
                truncation=True,
                max_length=256,
            )
        else:
            text_tok = tokenizer(
                [
                    s + "\n" + tokenizer.eos_token
                    for s in examples[TEXT_COLUMN]
                ],
                padding="do_not_pad",  # options: "longest", "max_length", "do_not_pad"
                truncation=False,
            )
        new_examples[TEXT_TOKENIZED_COLUMN] = text_tok["input_ids"]
        new_examples[TEXT_ATTN_MASK_COLUMN] = text_tok["attention_mask"]

        if TARGET_COLUMN in examples:
            target_tok = tokenizer(
                examples[TARGET_COLUMN],
                # truncation=True,
                padding="do_not_pad",
                # max_length=256,
            )
            new_examples[TARGET_TOKENIZED_COLUMN] = target_tok["input_ids"]
            new_examples[TARGET_ATTN_MASK_COLUMN] = target_tok["attention_mask"]
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
                TEXT_TOKENIZED_COLUMN,
                TEXT_ATTN_MASK_COLUMN,
                TARGET_TOKENIZED_COLUMN,
                TARGET_ATTN_MASK_COLUMN,
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

        # labels = result["input_ids"].copy()
        # if padding == "max_length" and ignore_pad_token_for_loss:
        #     labels = [
        #         [(id if id != tokenizer.pad_token_id else -100) for id in label]
        #         for label in labels
        #     ]
        # result["labels"] = labels

        return result

    return preprocessor


def preprocess_dataset(
    raw_dataset: DatasetDict,
    split: str,
    tokenizer,
    text_column: str,
    target_column: str | None = None,
    join_text_target: ColumnJoiner | None = None,
    max_sample_count: int | None = None,
    batch_size: int = 32,
    chunk_size: int | None = 256,
):
    do_chunk_text = chunk_size is not None and chunk_size > 0
    if split not in raw_dataset:
        raise ValueError(f"Split '{split}' not found in the dataset.")
    dataset = raw_dataset[split]
    if max_sample_count is not None:
        max_sample_count = min(len(dataset), max_sample_count)
        dataset = dataset.select(range(max_sample_count))

    dataset = dataset.map(
        get_column_preprocessor(
            text_column=text_column,
            target_column=target_column,
            join_text_target=join_text_target,
        ),
        batched=True,
        batch_size=batch_size,
    )

    # print(dataset[0][text_column])
    # print(dataset[0].get(target_column, "No target column"))
    # print(dataset[0].get("TEXT", "No TEXT column"))
    # print(dataset[0].get("TARGET", "No TARGET column"))

    dataset = dataset.map(
        get_tokenizer_preprocessor(tokenizer, truncate=not do_chunk_text),
        batched=True,
        batch_size=batch_size,
        remove_columns=[
            k
            for k in dataset.column_names
            if k
            not in [
                TEXT_TOKENIZED_COLUMN,
                TEXT_ATTN_MASK_COLUMN,
                TARGET_TOKENIZED_COLUMN,
                TARGET_ATTN_MASK_COLUMN,
            ]
        ],
    )
    # print(dataset[0:1])
    # print(f"Sample count after grouping: {len(dataset)}")
    for i in range(0):
        print(f"Example {i}:")
        print(f"Token count: {len(dataset[i][TEXT_TOKENIZED_COLUMN])}")
        print(
            tokenizer.decode(
                dataset[i][TEXT_TOKENIZED_COLUMN], skip_special_tokens=False
            )
        )

    if do_chunk_text and chunk_size is not None:
        dataset = dataset.map(
            get_group_text_preprocessor(block_size=chunk_size),
            batched=True,
            batch_size=batch_size,
        )
        # print(dataset[0:10])
        # print(f"Sample count after grouping: {len(dataset)}")
        for i in range(4):
            print(f"Example {i}:")
            print(f"Token count: {len(dataset[i][TEXT_TOKENIZED_COLUMN])}")
            print(
                tokenizer.decode(
                    dataset[i][TEXT_TOKENIZED_COLUMN], skip_special_tokens=False
                )
            )

    total_token_count = sum(
        len(dataset[i][TEXT_TOKENIZED_COLUMN]) for i in range(len(dataset))
    )
    print(f"{split} total token count: {total_token_count}")

    return dataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/amirreza/projects/ai/models/llm/llama-3.2-3B"
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset_name_or_path = "~/projects/ai/data/CodeSearchNet/python"
    # dataset_name_or_path = "~/projects/ai/data/spp_30k/SPP_30k_verified.jsonl"
    raw_dataset = load_raw_dataset(
        dataset_name_or_path,
        train_file="train.jsonl",
        test_file="test.jsonl",
        validation_file="valid.jsonl",
    )
    print(raw_dataset)
    print(raw_dataset["train"].column_names)

    text_column = "code"
    target_column = "docstring"

    def join_text_target(text: str, target: str) -> str:
        return f"{text}\nExplanation:\n{target}"

    train_dataset = preprocess_dataset(
        raw_dataset,
        "train",
        tokenizer=tokenizer,
        text_column=text_column,
        target_column=target_column,
        join_text_target=join_text_target,
        max_sample_count=10,
        chunk_size=256,
    )

    print(train_dataset.column_names)
