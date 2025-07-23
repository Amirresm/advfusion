from dataclasses import dataclass
from typing import Literal

import adapters
import peft


@dataclass
class PeftConfigs:
    LoRA = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        # task_type=peft.TaskType.CAUSAL_LM,
        task_type=None,
    )


@dataclass
class AdaptersConfigsClass:
    SeqBn = adapters.SeqBnConfig()
    LoRA = adapters.LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.1,
        selfattn_lora=True,
        intermediate_lora=False,
        output_lora=False,
        attn_matrices=["q", "v"],
    )


AdaptersConfigs = AdaptersConfigsClass()


def get_peft_config(peft_method: str, lib: Literal["peft", "adp"]):
    if lib == "peft":
        if peft_method == "lora":
            return PeftConfigs.LoRA
        else:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")
    elif lib == "adp":
        if peft_method == "seq_bn":
            return AdaptersConfigs.SeqBn
        elif peft_method == "lora":
            return AdaptersConfigs.LoRA
        else:
            raise ValueError(f"Unsupported Adapters method: {peft_method}")
    else:
        raise ValueError(f"Unsupported library: {lib}")
