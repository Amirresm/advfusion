from collections import defaultdict


def count_all_parameters(model):
    total = 0
    trainable = 0
    dtype_summary = defaultdict(int)

    for name, module in model.named_modules():
        total += sum(p.numel() for p in module.parameters())
        trainable += sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        for p in module.parameters():
            if str(p.dtype) not in dtype_summary:
                dtype_summary[str(p.dtype)] = 0
            dtype_summary[str(p.dtype)] += p.numel()

        # if hasattr(module, "weight") and isinstance(
        #     module.weight, torch.Tensor
        # ):
        #     numel = module.weight.numel()
        #     total += numel
        #     dtype_summary[str(module.weight.dtype)] += numel
        #     if module.weight.requires_grad:
        #         trainable += numel
        #
        # # If it's a quantized module like bnb.nn.Linear4bit, manually access .weight
        # elif hasattr(module, "weight") and hasattr(module.weight, "data"):
        #     numel = module.weight.data.numel()
        #     total += numel
        #     dtype_summary[str(module.weight.data.dtype)] += numel

    return total, trainable, dtype_summary


def report_model(model):
    print("=" * 40, "Model Structure:")
    print(model)

    print("=" * 40, "Model Configuration:")
    print(model.config)

    print("=" * 40, "Parameters:")
    for name, param in model.named_parameters():
        trainable = param.requires_grad
        is_lora = "lora" in name.lower() or "adapter" in name.lower()
        print(
            f"{name.replace("model.layers.", ""):<46s} | {str(param.dtype).replace("torch.",""):<10s} {str(param.shape).replace("torch.",""):<20s} | LoRA: {is_lora:<2} | trainable: {trainable:<2}"
        )

    # Verifying the datatypes.
    print("=" * 40, "Parameter Dtypes:")
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    total_params = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")

    total, trainable, dtypes = count_all_parameters(model)

    print(f"Total params (including quantized): {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("Parameter count by dtype:")
    for dtype, count in dtypes.items():
        print(f"  {dtype:<10}: {count:,}")
