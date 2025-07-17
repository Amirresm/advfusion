import os
from typing import Literal

import adapters
import peft
import torch

from src.peft.adapters_interfaces import get_adapter_interface


def setup_for_peft(
    model, peft_lib: Literal["adp", "peft"], config, dtype=torch.bfloat16
):
    if peft_lib == "adp":
        print("Using adapters: Initializing adapters")
        interface = get_adapter_interface(model.config.model_category)
        adapters.init(model, interface=interface)
        adapter_name = f"test_adapter"
        model.add_adapter(adapter_name, config=config)
        model.set_active_adapters(adapter_name)
        model.train_adapter(adapter_name, train_embeddings=True)
        model.adapter_to(adapter_name, device=model.device, dtype=dtype)

        # fusion_adapter_names = [adapter_name]
        # fusion_name = adapters.composition.Fuse(*fusion_adapter_names)
        # model.add_adapter_fusion(fusion_name, set_active=True)
        # model.adapter_fusion_to(fusion_name, device=model.device, dtype=dtype)
        # model.train_adapter_fusion(fusion_name)
        # print(f"Adapter {adapter_name} moved to device {model.device} with dtype {torch.bfloat16}")

        print(model.active_adapters)
        print(model.adapters_config.adapters)

        # for param in model.parameters():
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.data = param.data.to(torch.float32)
        #
        # class CastOutputToFloat(torch.nn.Sequential):
        #     def forward(self, x): return super().forward(x).to(torch.float32)
        # model.lm_head = CastOutputToFloat(model.lm_head)
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
    adapter_name = f"test_adapter"
    if peft_lib == "adp":
        interface = get_adapter_interface(model.config.model_category)
        adapters.init(model, interface=interface)
        model.add_adapter(adapter_name, config=config)
        model.set_active_adapters(adapter_name)
        model.train_adapter(adapter_name, train_embeddings=True)
        model.adapter_to(adapter_name, device=model.device, dtype=dtype)
        model.load_adapter(
            os.path.join(peft_path, adapter_name),
            load_as=adapter_name,
            set_active=True,
        )
    elif peft_lib == "peft":
        model = peft.PeftModel.from_pretrained(
            model,
            peft_path,
            adapter_name=adapter_name,
            is_trainable=True,
            # torch_device=model.device,
        )

    return model
