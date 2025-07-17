import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "model_name_or_path",
        type=str,
        help="Path to the model or model name.",
    )
    args.add_argument(
        "--q",
        "--quantization_mode",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode to use for the model.",
    )
    args.add_argument(
        "--peft", type=str, default="lora", help="PEFT method to use."
    )
    args.add_argument(
        "--lib",
        type=str,
        default="peft",
        choices=["peft", "adp"],
        help="Library to use for PEFT methods.",
    )

    args = args.parse_args()

    from src.model.report import report_model
    from src.model.model import init_model, init_tokenizer
    from src.peft.configs import get_peft_config
    from src.peft.peft import setup_for_peft

    model, model_dtype = init_model(
        args.model_name_or_path,
        quantization_mode=args.q,
    )
    tokenizer = init_tokenizer(args.model_name_or_path, model)

    print(f"Model name: {model.config.model_name}")
    print(f"Model category: {model.config.model_category}")
    print(f"Model loaded with dtype: {model_dtype}")

    if args.q is not None:
        peft_config = get_peft_config(args.peft, args.lib)
        model = setup_for_peft(
            model, args.lib, config=peft_config, dtype=model_dtype
        )

    # report_model(model)
