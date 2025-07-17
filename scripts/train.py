import argparse

import torch

from src.model.report import report_model
from src.generation.generation import generate_raw_samples
from src.model.model import init_model, init_tokenizer
from src.dataset.dataset import load_raw_dataset, preprocess_dataset
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
        "--input_column",
        type=str,
        default=None,
        help="Column name for the input data in the dataset.",
    )
    args.add_argument(
        "--target_column",
        type=str,
        default=None,
        help="Column name for the target data in the dataset.",
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

    model, model_dtype = init_model(
        args.model_name_or_path,
        quantization_mode=args.q,
    )
    tokenizer = init_tokenizer(args.model_name_or_path, model)

    raw_dataset = load_raw_dataset(
        args.dataset_name_or_path,
        train_file=args.train_file,
        test_file=args.test_file,
        validation_file=args.validation_file,
    )

    generate_raw_samples(
        model,
        tokenizer,
        raw_dataset["test"][:0],
        args.input_column,
        args.target_column,
        batch_size=1,
    )

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
    elif args.q is not None:
        peft_config = get_peft_config(args.peft, args.lib)
        model = setup_for_peft(
            model, args.lib, config=peft_config, dtype=model_dtype
        )

    def join_text_target(text: str, target: str) -> str:
        return f"{text}\nExplanation:\n{target}"

    train_dataset = preprocess_dataset(
        raw_dataset,
        "train",
        tokenizer=tokenizer,
        text_column=args.input_column,
        target_column=args.target_column,
        join_text_target=join_text_target,
        max_sample_count=args.max_train_samples,
        chunk_size=args.chunk_size,
    )
    eval_dataset = preprocess_dataset(
        raw_dataset,
        "validation",
        tokenizer=tokenizer,
        text_column=args.input_column,
        target_column=args.target_column,
        join_text_target=join_text_target,
        max_sample_count=args.max_eval_samples,
        chunk_size=args.chunk_size,
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
            eval_steps=500,
        )
        trainer.train()

        trainer.save_model()

    with torch.no_grad(), torch.amp.autocast("cuda"), torch.inference_mode():
        generate_raw_samples(
            model,
            tokenizer,
            raw_dataset["test"][: args.max_test_samples],
            args.input_column,
            args.target_column,
            batch_size=8,
        )


if __name__ == "__main__":
    main()
