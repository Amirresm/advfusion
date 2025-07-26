from dataclasses import dataclass, field
from typing import Literal

from simple_parsing import choice


@dataclass
class PeftArgs:
    lib: Literal["adp", "peft"] | None = field(
        metadata={
            "choices": ["adp", "peft"],
            "help": (
                "Library to use for PEFT methods. Options are 'peft' or 'adp'."
            ),
        },
    )
    peft_method: str | None = field(
        metadata={"help": "PEFT method to use."},
    )
    preload_peft_from: str | None = field(
        metadata={
            "help": "Path to the PEFT model to preload.",
        },
    )
