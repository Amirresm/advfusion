import argparse
import os

import torch

from src.dataset.custom_processors import csn_processor
from src.generation.generation import generate_raw_samples
from src.model.model import init_model, init_tokenizer
from src.dataset.dataset import load_raw_dataset, preprocess_dataset
from src.model.report import report_model
from src.peft.configs import get_peft_config
from src.peft.peft import load_peft, setup_for_peft
from src.train.train import get_trainer


def main():
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
        "--lib",
        type=str,
        default="peft",
        choices=["peft", "adp"],
        help="Library to use for PEFT methods.",
    )
    args.add_argument(
        "--peft", type=str, default=None, help="PEFT method to use."
    )
    args.add_argument(
        "--preload_peft_from",
        type=str,
        default=None,
        help="Path to the PEFT model to preload.",
    )

    args.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="Path to the dataset or dataset name.",
    )
    args.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="File name for the training dataset.",
    )
    args.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="File name for the validation dataset.",
    )
    args.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="File name for the test dataset.",
    )
    args.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use for training.",
    )
    args.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use for evaluation.",
    )
    args.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use for test.",
    )
    args.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Chunk size for the training dataset.",
    )
    args.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the model and training artifacts.",
    )

    args = args.parse_args()

    try:
        args.validation_file = float(args.validation_file)
    except ValueError:
        pass
    try:
        args.test_file = float(args.test_file)
    except ValueError:
        pass

    output_dir = os.path.realpath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model, model_dtype = init_model(
        args.model_name_or_path,
        quantization_mode=args.q,
    )
    tokenizer = init_tokenizer(args.model_name_or_path, model)

    print(
        f"Tokenizer padding token: {tokenizer.pad_token} ({tokenizer.pad_token_id})"
    )
    print(f"Model padding token: {model.config.pad_token_id}")
    print(
        f"Tokenizer eos token: {tokenizer.eos_token} ({tokenizer.eos_token_id})"
    )
    print(f"Model eos token: {model.config.eos_token_id}")

    raw_dataset = load_raw_dataset(
        args.dataset_name_or_path,
        raw_preprocessor=csn_processor,
        train_file=args.train_file,
        test_file=args.test_file,
        validation_file=args.validation_file,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_eval_samples,
        max_test_samples=args.max_test_samples,
        load_from_cache_file=False,
    )

    generate_raw_samples(
        model,
        tokenizer,
        raw_dataset["test"][:0],
        batch_size=1,
        save_path=os.path.join(args.output_dir, "pre_trained"),
    )

    if args.lib is not None:
        if args.preload_peft_from is not None:
            print(f"Loading PEFT model from {args.preload_peft_from}")
            peft_config = get_peft_config(args.peft, args.lib)
            model = load_peft(
                model,
                peft_lib=args.lib,
                peft_path=args.preload_peft_from,
                config=peft_config,
                dtype=model_dtype,
            )
        elif args.peft is not None:
            peft_config = get_peft_config(args.peft, args.lib)
            model = setup_for_peft(
                model, args.lib, config=peft_config, dtype=model_dtype
            )
        else:
            print(
                "Warning: Peft library is specified but no PEFT config is provided."
            )

    report_model(model)

    train_dataset = preprocess_dataset(
        raw_dataset,
        "train",
        tokenizer=tokenizer,
        only_completion=True,
        chunk_size=args.chunk_size,
        load_from_cache_file=False,
    )
    eval_dataset = preprocess_dataset(
        raw_dataset,
        "validation",
        tokenizer=tokenizer,
        only_completion=True,
        chunk_size=0,
        text_max_length=256,
        target_max_length=128,
        load_from_cache_file=False,
    )

    if args.do_train:
        print("Training the model...")
        trainer = get_trainer(
            model=model,
            tokenizer=tokenizer,
            peft_lib=args.lib,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            train_batch_size=4,
            eval_batch_size=4,
            epochs=1,
            logging_steps=100,
            eval_steps=200,
        )
        trainer.train()

        save_path = os.path.join(args.output_dir, "adapter")
        if args.lib == "adp":
            model.save_adapter(
                save_path, "adapter"
            )
        else:
            model.save_pretrained(save_path)

    with torch.inference_mode():
        generate_raw_samples(
            model,
            tokenizer,
            raw_dataset["test"][: args.max_test_samples],
            batch_size=4,
            save_path=os.path.join(args.output_dir, "fine_tuned"),
        )


if __name__ == "__main__":
    main()
