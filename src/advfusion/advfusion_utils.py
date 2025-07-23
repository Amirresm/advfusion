def get_module_modifier(device, adapter_name, freeze=False, zero=False):
    def processor(_, module):
        if hasattr(module, "adapters") and adapter_name in module.adapters:
            if freeze:
                module.adapters[adapter_name].adapter_down[
                    0
                ].weight.requires_grad = False
                module.adapters[adapter_name].adapter_down[
                    0
                ].bias.requires_grad = False

                module.adapters[
                    adapter_name
                ].adapter_up.weight.requires_grad = False
                module.adapters[adapter_name].adapter_up.bias.requires_grad = (
                    False
                )

            if zero:
                module.adapters[adapter_name].adapter_down[0].weight.data.fill_(
                    0
                )
                module.adapters[adapter_name].adapter_down[0].bias.data.fill_(0)

                module.adapters[adapter_name].adapter_up.weight.data.fill_(0)
                module.adapters[adapter_name].adapter_up.bias.data.fill_(0)

    return processor


def reload_adapter(model, adapter_path, adapter_name, dtype):
    print(f"Reloading adapter {adapter_name} from {adapter_path}")
    model.load_adapter(
        adapter_path,
        load_as=adapter_name,
        set_active=True,
    )
    model.adapter_to(adapter_name, device=model.device, dtype=dtype)


def freeze_adapter(model, adapter_name, freeze=True):
    processor = get_module_modifier(
        model.device, adapter_name, freeze=freeze, zero=False
    )
    model.apply_to_adapter_layers(processor)


def zero_adapter(model, adapter_name):
    processor = get_module_modifier(
        model.device, adapter_name, freeze=False, zero=True
    )
    model.apply_to_adapter_layers(processor)
