import os

import adapters
import torch

from src.advfusion.advfusion_utils import (
    zero_adapter,
    freeze_adapter,
    reload_adapter,
)
from src.peft.adapters_interfaces import get_adapter_interface

from rich import print


def init_advfusion(
    model,
    model_type,
    adapter_path_list,
    target_adapter_path,
    dtype=torch.bfloat16,
):
    interface = get_adapter_interface(model_type)
    print(f"Initializing model adapters with interface for {model_type}.")
    adapters.init(model, interface=interface)

    target_adapter_name = target_adapter_path.split("/")[-1]
    adapter_names = []
    for adapter_path in adapter_path_list:
        adapter_name = adapter_path.split("/")[-1]
        path = os.path.join(adapter_path, "adapter", "adapter")

        adapter_names.append(adapter_name)
        model.load_adapter(
            path,
            load_as=adapter_name,
            set_active=True,
        )
        model.adapter_to(adapter_name, device=model.device, dtype=dtype)

        freeze_adapter(
            model,
            adapter_name=adapter_name,
            freeze=True,
        )

    zero_adapter(
        model,
        adapter_name=target_adapter_name,
    )

    all_adapter_names = adapter_names
    model.set_active_adapters(all_adapter_names)

    fusion_name = adapters.composition.Fuse(*all_adapter_names)
    model.add_adapter_fusion(fusion_name, set_active=True)
    model.adapter_fusion_to(fusion_name, device=model.device, dtype=dtype)
    model.train_adapter_fusion(fusion_name)

    print("Adapter Config:", model.adapters_config.__dict__)
    print("Active Adapters:", model.active_adapters)
    print(model.adapter_summary())

    return fusion_name


def load_advfusion(
    model,
    model_type,
    target_adapter_path,
    preload_path,
    dtype=torch.bfloat16,
):
    interface = get_adapter_interface(model_type)
    print(f"Initializing model adapters with interface for {model_type}.")
    adapters.init(model, interface=interface)

    target_adapter_name = target_adapter_path.split("/")[-1]
    preload_path_advf = os.path.join(preload_path, "adapter_fusion")
    fusion_name, _ = model.load_adapter_setup(
        preload_path_advf, set_active=True, use_safetensors=True
    )
    print("Fusion:", fusion_name, type(fusion_name))

    adapter_names = [
        k for k in model.adapters_config.adapters.keys() if k != fusion_name
    ]
    model.set_active_adapters(fusion_name)
    model.adapter_fusion_to(fusion_name, device=model.device, dtype=dtype)
    for adapter_name in adapter_names:
        model.adapter_to(adapter_name, device=model.device, dtype=dtype)
        freeze_adapter(
            model,
            adapter_name=adapter_name,
            freeze=True,
        )
    zero_adapter(
        model,
        adapter_name=target_adapter_name,
    )
    model.train_adapter_fusion(fusion_name)

    print("Adapter Config:", model.adapters_config.__dict__)
    print("Active Adapters:", model.active_adapters)
    print(model.adapter_summary())

    return fusion_name


def reload_advf_target_adapter(
    model, target_adapter_path, dtype=torch.bfloat16
):
    adapter_name = target_adapter_path.split("/")[-1]
    path = os.path.join(target_adapter_path, "adapter", "adapter")

    reload_adapter(
        model,
        adapter_path=path,
        adapter_name=adapter_name,
        dtype=dtype,
    )
    freeze_adapter(
        model,
        adapter_name=adapter_name,
        freeze=True,
    )
