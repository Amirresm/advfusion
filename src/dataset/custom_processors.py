from typing import Callable, TypedDict

from src.dataset.utils import DatasetType


class RawRow(TypedDict):
    TEXT: list[str]
    TARGET: list[str] | None


type RawPreprocessor = Callable[[dict[str, list]], RawRow]


def ct_processor(example: dict) -> RawRow:
    """Code translation dataset processor."""
    prompts = []
    targets = []
    for i in range(len(example["id"])):
        source_lang = example["source_lang"][i]
        target_lang = example["target_lang"][i]

        prompt = example["source_content"][i]
        target = example["target_content"][i]

        prompt = f"### Code written in {source_lang}:\n{prompt}\n### Code written in {target_lang}:\n"
        prompts.append(prompt)
        targets.append(target)

    return {
        "TEXT": prompts,
        "TARGET": targets,
    }


def csn_processor(example: dict) -> RawRow:
    prompts = []
    targets = []
    for i in range(len(example["code"])):
        prompt = example["code"][i]
        target = (
            example.get("docstring", [""])[i] if "docstring" in example else ""
        )
        if target:
            prompt = prompt.replace(target, "")

        prompt = f"{prompt}\n### Response:\n"
        prompts.append(prompt)
        targets.append(target)

    return {
        "TEXT": prompts,
        "TARGET": targets,
    }


def code_gen_processor(example: dict) -> RawRow:
    prompts = []
    targets = []
    for i in range(len(example["code"])):
        code = example["code"][i]
        s = code.split('"""\n')
        prompt = "".join(s[:-1]) + '"""\n'
        target = s[-1] if len(s) > 1 else ""

        prompts.append(prompt)
        if target:
            targets.append(target)
    return {
        "TEXT": prompts,
        "TARGET": targets,
    }


def get_dataset_processor(dataset_type: DatasetType) -> RawPreprocessor:
    if dataset_type == DatasetType.CodeSearchNet:
        return csn_processor
    elif dataset_type == DatasetType.CodeGeneration:
        return code_gen_processor
    elif dataset_type == DatasetType.CodeTranslation:
        return ct_processor
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
