from typing import Callable, TypedDict


class RawRow(TypedDict):
    TEXT: list[str]
    TARGET: list[str] | None


type RawPreprocessor = Callable[[dict[str, list]], RawRow]


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

        prompt.replace(target, "")
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
