import json
import os

from datasets import DatasetDict, load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from rich import print

from src.dataset.custom_processors import (
    RawPreprocessor,
    get_dataset_processor,
)
from src.dataset.processors import (
    ChunkTextPreprocessor,
    TokenizeTextPreprocessor,
)
from src.dataset.utils import DatasetType


def visualize_rows(tokenizer, rows):
    for i in range(len(rows["input_ids"])):
        input_ids = rows["input_ids"][i]
        labels = rows["labels"][i]

        print(f"Example {i}:")
        print(f"Token count: {len(input_ids)}")
        for j in range(len(input_ids)):
            id = input_ids[j]
            token = tokenizer.decode(id, skip_special_tokens=False)
            label = labels[j]
            color = "green" if label != -100 else "red"
            print(f"[{color}]{token}[/]", end="")
        print()


def get_raw_preprocessor(
    processor: RawPreprocessor,
):
    def preprocessor(examples):
        return {
            **examples,
            **processor(examples),
        }

    return preprocessor


def load_raw_dataset(
    dataset_path: str,
    dataset_type: DatasetType,
    train_file: str | None = None,
    test_file: str | float = 0.2,
    validation_file: str | float = 0.5,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_test_samples: int | None = None,
    load_from_cache_file: bool = True,
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

    if (
        "test" not in raw_dataset
        and isinstance(test_file, float)
        and test_file > 0
    ):
        print("DATASET: Splitting train dataset into test and validation sets.")
        raw_dataset = raw_dataset["train"].train_test_split(
            test_size=test_file, seed=42
        )
    if (
        "validation" not in raw_dataset
        and isinstance(validation_file, float)
        and validation_file > 0
    ):
        print("DATASET: Splitting train dataset into validation and test sets.")
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

    for split in raw_dataset.keys():
        length = len(raw_dataset[split])
        print(f"DATASET: Loaded split '{split}' with {length:,} samples.")

    if max_train_samples is not None:
        raw_dataset["train"] = raw_dataset["train"].select(
            range(min(max_train_samples, len(raw_dataset["train"])))
        )
    if max_validation_samples is not None and "validation" in raw_dataset:
        raw_dataset["validation"] = raw_dataset["validation"].select(
            range(min(max_validation_samples, len(raw_dataset["validation"])))
        )
    if max_test_samples is not None and "test" in raw_dataset:
        raw_dataset["test"] = raw_dataset["test"].select(
            range(min(max_test_samples, len(raw_dataset["test"])))
        )

    for split in raw_dataset.keys():
        length = len(raw_dataset[split])
        print(
            f"DATASET: After limiting, split '{split}' has {length:,} samples."
        )

    raw_preprocessor = get_dataset_processor(dataset_type)
    for split, dataset in raw_dataset.items():
        raw_dataset[split] = dataset.map(
            get_raw_preprocessor(raw_preprocessor),
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            load_from_cache_file=load_from_cache_file,
        )
        print(f"DATASET: columns: {raw_dataset[split].column_names}")
    return raw_dataset


def preprocess_dataset(
    raw_dataset: DatasetDict,
    split: str,
    tokenizer,
    output_dir: str,
    max_sample_count: int | None = None,
    batch_size: int = 32,
    only_completion: bool = False,
    chunk_size: int | None = 256,
    text_max_length: int | None = 256,
    target_max_length: int | None = 128,
    load_from_cache_file=True,
):
    do_chunk_text = chunk_size is not None and chunk_size > 0
    if split not in raw_dataset:
        raise ValueError(f"Split '{split}' not found in the dataset.")
    dataset = raw_dataset[split]

    tokenizer_preprocessor = TokenizeTextPreprocessor(
        tokenizer,
        truncate=not do_chunk_text,
        text_max_length=text_max_length,
        target_max_length=target_max_length,
        only_completion=only_completion,
    )
    dataset = dataset.map(
        tokenizer_preprocessor.get_tokenizer_preprocessor(),
        batched=True,
        batch_size=batch_size,
        remove_columns=[
            k
            for k in dataset.column_names
            if k
            not in [
                "input_ids",
                "attention_mask",
                "labels",
            ]
        ],
        load_from_cache_file=load_from_cache_file,
    )
    visualize_rows(tokenizer, dataset[:3])

    if do_chunk_text and chunk_size is not None:
        chunk_text_preprocessor = ChunkTextPreprocessor(chunk_size)
        dataset = dataset.map(
            chunk_text_preprocessor.get_group_text_preprocessor(),
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=load_from_cache_file,
        )
        chunk_text_preprocessor.report()
        visualize_rows(tokenizer, dataset[:3])

    if max_sample_count is not None:
        max_sample_count = min(len(dataset), max_sample_count)
        dataset = dataset.select(range(max_sample_count))

    total_token_count = sum(
        sum([1 if label != -100 else 0 for label in dataset[i]["labels"]])
        for i in range(len(dataset))
    )
    print(
        f"DATASET (PROCESSED): '{split}' total rows {len(dataset):,}, total token count: {total_token_count:,}"
    )

    metadata = {
        "split": split,
        "max_sample_count": max_sample_count,
        "chunk_size": chunk_size,
        "text_max_length": text_max_length if not chunk_size else None,
        "target_max_length": target_max_length if not chunk_size else None,
        "only_completion": only_completion,
        "total_token_count": total_token_count,
    }

    with open(os.path.join(output_dir, f"ds_{split}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return dataset


def pad_seq(
    tokenizer,
    sequences: list[str],
    should_pad: bool = False,
    max_length: int | None = None,
    side: str = "right",
):
    if not should_pad:
        return sequences
    tokenized = tokenizer(
        sequences,
        padding="longest",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        padding_side=side,
    )

    return tokenizer.batch_decode(
        tokenized["input_ids"], skip_special_tokens=True
    )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/storage/ai/models/llm/codegemma-2b"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = tokenizer.special_tokens_map

    print(special_tokens)

    dataset_name_or_path = "~/projects/ai/data/CodeSearchNet/python"
    dataset_name_or_path = "~/projects/ubc/ct_dataset/processed_data/rust"
    # dataset_name_or_path = "~/projects/ai/data/spp_30k/SPP_30k_verified.jsonl"

    dataset_type = DatasetType.get_dataset_type(dataset_name_or_path)

    print(f"Dataset type: {dataset_type}")

    raw_dataset = load_raw_dataset(
        dataset_name_or_path,
        dataset_type=dataset_type,
        train_file="train.jsonl",
        test_file="test.jsonl",
        validation_file="valid.jsonl",
        # test_file=50,
        # validation_file=0,
        # max_train_samples=50,
        # max_validation_samples=5,
        # max_test_samples=100,
        load_from_cache_file=False,
    )
    # print(raw_dataset)
    # print(raw_dataset["train"].column_names)
    # print(raw_dataset["train"][0][TEXT_COLUMN])
    # print(raw_dataset["train"][0][TARGET_COLUMN])

    train_dataset = preprocess_dataset(
        raw_dataset,
        "train",
        tokenizer=tokenizer,
        # max_sample_count=50,
        only_completion=False,
        chunk_size=512,
        load_from_cache_file=False,
    )
    # eval_dataset = preprocess_dataset(
    #     raw_dataset,
    #     "validation",
    #     tokenizer=tokenizer,
    #     max_sample_count=1,
    #     only_completion=True,
    #     chunk_size=512,
    #     text_max_length=256,
    #     target_max_length=256,
    #     # batch_size=1,
    #     load_from_cache_file=False,
    # )
