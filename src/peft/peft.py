import os
from typing import Literal

import adapters
import peft
import torch
from rich import print

from src.peft.adapters_interfaces import get_adapter_interface


def setup_for_peft(
    model, peft_lib: Literal["adp", "peft"], config, dtype=torch.bfloat16
):
    if peft_lib == "adp":
        print("Using adapters: Initializing adapters")
        interface = get_adapter_interface(model.config.model_category)
        print(
            f"Initializing model adapters with interface for {model.config.model_category}."
        )
        adapters.init(model, interface=interface)

        adapter_name = f"adapter"
        model.add_adapter(adapter_name, config=config)
        model.adapter_to(adapter_name, device=model.device, dtype=dtype)
        model.set_active_adapters(adapter_name)
        model.train_adapter(adapter_name, train_embeddings=True)

        # for param in model.parameters():
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.data = param.data.to(torch.float32)
        #
        # class CastOutputToFloat(torch.nn.Sequential):
        #     def forward(self, x): return super().forward(x).to(torch.float32)
        # model.lm_head = CastOutputToFloat(model.lm_head)

        print("Adapter Config:", model.adapters_config.__dict__)
        print("Active Adapters:", model.active_adapters)
        print(model.adapter_summary())
    elif peft_lib == "peft":
        print("Using PEFT: Initializing PEFT")
        model = peft.get_peft_model(
            model,
            config,
        )
        model.print_trainable_parameters()

    return model


def load_peft(
    model,
    peft_lib: Literal["adp", "peft"],
    peft_path,
    config,
    dtype=torch.bfloat16,
):
    adapter_name = f"adapter"
    if peft_lib == "adp":
        print("Using adapters: Initializing adapters")
        interface = get_adapter_interface(model.config.model_category)
        print(
            f"Initializing model adapters with interface for {model.config.model_category}."
        )
        adapters.init(model, interface=interface)
        # model.add_adapter(adapter_name, config=config)
        model.load_adapter(
            os.path.join(peft_path, adapter_name),
            load_as=adapter_name,
            set_active=True,
        )
        model.adapter_to(adapter_name, device=model.device, dtype=dtype)
        model.set_active_adapters(adapter_name)
        model.train_adapter(adapter_name, train_embeddings=True)

        print("Adapter Config:", model.adapters_config.__dict__)
        print("Active Adapters:", model.active_adapters)
        print(model.adapter_summary())
    elif peft_lib == "peft":
        print("Using PEFT: Loading PEFT model")
        model = peft.PeftModel.from_pretrained(
            model,
            os.path.join(peft_path, adapter_name),
            adapter_name=adapter_name,
            is_trainable=True,
            # torch_device=model.device,
        )

    return model
