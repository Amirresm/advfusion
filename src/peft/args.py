from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PeftArgs:
    """Peft arguments for model fine-tuning."""

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
