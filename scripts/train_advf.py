import argparse
import os

import torch

from src.advfusion.advfusion import (
    init_advfusion,
    load_advfusion,
    reload_advf_target_adapter,
)
from src.dataset.custom_processors import csn_processor
from src.dataset.utils import DatasetType
from src.generation.generation import generate_raw_samples
from src.model.model import init_model, init_tokenizer
from src.dataset.dataset import load_raw_dataset, preprocess_dataset
from src.model.report import report_model
from src.model.utils import ModelType
from src.train.trainer import get_trainer


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "model_name_or_path",
        type=str,
        help="Path to the model or model name.",
    )
    args.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Type of the model to use (e.g., 'llama-3', 'codellama'). Leave empty to auto-detect.",
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
        "--adapter_path_list",
        type=str,
        default=None,
        help="Comma-separated list of adapter paths to load.",
    )
    args.add_argument(
        "--target_adapter_path",
        type=str,
        default=None,
        help="Path to the target adapter for AdvFusion.",
    )
    args.add_argument(
        "--preload_advf_from",
        type=str,
        default=None,
        help="Path to the AdvFusion adapter to preload.",
    )

    args.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="Path to the dataset or dataset name.",
    )
    args.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        help="Type of the dataset to use (e.g., 'codesearchnet', 'codegeneration'). Leave empty to auto-detect.",
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

    args.adapter_path_list = [
        p
        for p in (
            args.adapter_path_list.split(",") if args.adapter_path_list else []
        )
        if p.strip()
    ]

    try:
        args.validation_file = float(args.validation_file)
    except ValueError:
        pass
    try:
        args.test_file = float(args.test_file)
    except ValueError:
        pass

    args.model_type = (
        ModelType.get_model_type(args.model_name_or_path)
        if args.model_type is None
        else ModelType.get_model_type(args.model_type)
    )
    args.dataset_type = (
        DatasetType.get_dataset_type(args.dataset_name_or_path)
        if args.dataset_type is None
        else DatasetType.get_dataset_type(args.dataset_type)
    )

    output_dir = os.path.realpath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model, model_dtype = init_model(
        args.model_name_or_path,
        quantization_mode=args.q,
    )
    tokenizer = init_tokenizer(args.model_name_or_path, model)

    raw_dataset = load_raw_dataset(
        args.dataset_name_or_path,
        dataset_type=args.dataset_type,
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

    if args.preload_advf_from is not None:
        print(f"Loading AdvFusion model from {args.preload_advf_from}")
        fusion_name = load_advfusion(
            model,
            adapter_path_list=args.adapter_path_list,
            target_adapter_path=args.target_adapter_path,
            preload_path=args.preload_advf_from,
            dtype=model_dtype,
        )
    else:
        fusion_name = init_advfusion(
            model,
            adapter_path_list=args.adapter_path_list,
            target_adapter_path=args.target_adapter_path,
            dtype=model_dtype,
        )

    train_dataset = preprocess_dataset(
        raw_dataset,
        "train",
        tokenizer=tokenizer,
        output_dir=output_dir,
        only_completion=False,
        chunk_size=args.chunk_size,
        load_from_cache_file=False,
    )
    eval_dataset = preprocess_dataset(
        raw_dataset,
        "validation",
        tokenizer=tokenizer,
        output_dir=output_dir,
        only_completion=True,
        chunk_size=0,
        text_max_length=512,
        target_max_length=512,
        load_from_cache_file=False,
    )

    if args.do_train:
        print("Training the model...")
        trainer = get_trainer(
            model=model,
            tokenizer=tokenizer,
            peft_lib="adp",
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            train_batch_size=2,
            eval_batch_size=4,
            epochs=1,
            logging_steps=100,
            eval_steps=200,
        )
        trainer.train()

        with torch.inference_mode():
            generate_raw_samples(
                model,
                tokenizer,
                raw_dataset["test"][: args.max_test_samples],
                batch_size=4,
                save_path=os.path.join(args.output_dir, "excluded"),
            )

        reload_advf_target_adapter(
            model,
            target_adapter_path=args.target_adapter_path,
            dtype=model_dtype,
        )
        model.train_adapter_fusion(fusion_name)
        trainer.train()

        # trainer.save_model()
        # model.save_adapter_fusion(args.output_dir, fusion_name)
        fusion_path = os.path.join(args.output_dir, "adapter_fusion")
        model.save_adapter_setup(fusion_path, fusion_name)
        # all_fusions_path = os.path.join(args.output_dir, "all_adapter_fusions")
        # model.save_all_adapter_fusions(all_fusions_path)
    else:
        with torch.inference_mode():
            generate_raw_samples(
                model,
                tokenizer,
                raw_dataset["test"][: args.max_test_samples],
                batch_size=4,
                save_path=os.path.join(args.output_dir, "excluded"),
            )

        reload_advf_target_adapter(
            model,
            target_adapter_path=args.target_adapter_path,
            dtype=model_dtype,
        )

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
