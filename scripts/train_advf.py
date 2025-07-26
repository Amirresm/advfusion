from dataclasses import dataclass
import json
import os

import torch
import simple_parsing
from rich import print as rprint

from src.advfusion.advfusion import (
    init_advfusion,
    load_advfusion,
    reload_advf_target_adapter,
)
from src.advfusion.args import AdvfArgs
from src.dataset.args import DatasetArgs
from src.dataset.utils import DatasetType
from src.generation.args import GenArgs
from src.generation.generation import generate_raw_samples
from src.model.args import ModelArgs
from src.model.model import init_model, init_tokenizer
from src.dataset.dataset import load_raw_dataset, preprocess_dataset
from src.model.utils import ModelType
from src.train.args import TrainArgs
from src.train.trainer import get_trainer


@dataclass
class Args:
    model: ModelArgs
    advf: AdvfArgs
    dataset: DatasetArgs
    train: TrainArgs
    gen: GenArgs

    output_dir: str = (
        ""  # The output directory where the model predictions and checkpoints will be written.
    )


def main():
    args = simple_parsing.parse(Args)
    assert type(args.model.model_type) is ModelType
    assert type(args.dataset.dataset_type) is DatasetType

    output_dir = os.path.realpath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_dir = os.path.join(output_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model, model_dtype = init_model(
        args.model.model_name_or_path,
        quantization_mode=args.model.q,
    )
    tokenizer = init_tokenizer(args.model.model_name_or_path, model)

    raw_dataset = load_raw_dataset(
        args.dataset.dataset_name_or_path,
        dataset_type=args.dataset.dataset_type,
        train_file=args.dataset.train_file,
        test_file=args.dataset.test_file,
        validation_file=args.dataset.validation_file,
        max_train_samples=args.dataset.max_train_samples,
        max_validation_samples=args.dataset.max_eval_samples,
        max_test_samples=args.dataset.max_test_samples,
    )

    generate_raw_samples(
        model,
        tokenizer,
        raw_dataset,
        args.gen.gen_pre_train_max_samples,
        batch_size=args.gen.gen_batch_size,
        max_new_tokens=args.gen.generate_max_new_tokens,
        save_path=os.path.join(results_dir, "pre_trained"),
    )

    if args.advf.preload_advf_from is not None:
        print(f"Loading AdvFusion model from {args.advf.preload_advf_from}")
        fusion_name = load_advfusion(
            model,
            adapter_path_list=args.advf.adapter_path_list,
            target_adapter_path=args.advf.target_adapter_path,
            preload_path=args.advf.preload_advf_from,
            dtype=model_dtype,
        )
    else:
        fusion_name = init_advfusion(
            model,
            adapter_path_list=args.advf.adapter_path_list,
            target_adapter_path=args.advf.target_adapter_path,
            dtype=model_dtype,
        )

    train_dataset = None
    if "train" in raw_dataset:
        train_dataset = preprocess_dataset(
            raw_dataset,
            "train",
            tokenizer=tokenizer,
            output_dir=output_dir,
            only_completion=False,
            chunk_size=args.dataset.chunk_size,
            load_from_cache_file=False,
        )
    eval_dataset = None
    if "validation" in raw_dataset:
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
    test_dataset = None
    if "test" in raw_dataset:
        test_dataset = preprocess_dataset(
            raw_dataset,
            "test",
            tokenizer=tokenizer,
            output_dir=output_dir,
            only_completion=True,
            chunk_size=0,
            text_max_length=512,
            target_max_length=512,
            load_from_cache_file=False,
        )

    if train_dataset and args.train.do_train:
        print("Training the model...")
        trainer = get_trainer(
            model=model,
            peft_lib="adp",
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            train_batch_size=args.train.train_batch_size,
            eval_batch_size=args.train.eval_batch_size,
            epochs=args.train.epochs,
            learning_rate=args.train.learning_rate,
            warmup_ratio=args.train.warmup_ratio,
            logging_steps=args.train.logging_steps,
            eval_steps=args.train.eval_steps,
        )
        trainer.train()

        with torch.inference_mode():
            generate_raw_samples(
                model,
                tokenizer,
                raw_dataset,
                args.gen.gen_pre_train_max_samples,
                batch_size=args.gen.gen_batch_size,
                max_new_tokens=args.gen.generate_max_new_tokens,
                save_path=os.path.join(results_dir, "excluded"),
            )

        reload_advf_target_adapter(
            model,
            target_adapter_path=args.advf.target_adapter_path,
            dtype=model_dtype,
        )
        model.train_adapter_fusion(fusion_name)
        trainer.train()

        fusion_path = os.path.join(args.output_dir, "adapter_fusion")
        model.save_adapter_setup(fusion_path, fusion_name)

        if args.train.do_eval:
            with torch.inference_mode():
                eval_results = {}
                if eval_dataset:
                    print("Evaluating on the validation set...")
                    eval_results = trainer.evaluate(
                        eval_dataset=eval_dataset,  # type: ignore
                        metric_key_prefix="eval",
                    )

                test_results = {}
                if test_dataset:
                    print("Evaluating on the test set...")
                    test_results = trainer.evaluate(
                        eval_dataset=test_dataset,  # type: ignore
                        metric_key_prefix="test",
                    )

            rprint(f"Evaluation results:", eval_results)
            rprint(f"Test results:", test_results)

            evaluation_results = {
                "eval": eval_results,
                "test": test_results,
            }
            with open(
                os.path.join(results_dir, "evaluation_results.json"),
                "w",
            ) as f:
                json.dump(evaluation_results, f, indent=4)
    else:
        with torch.inference_mode():
            generate_raw_samples(
                model,
                tokenizer,
                raw_dataset,
                args.gen.gen_pre_train_max_samples,
                batch_size=args.gen.gen_batch_size,
                max_new_tokens=args.gen.generate_max_new_tokens,
                save_path=os.path.join(results_dir, "excluded"),
            )

        reload_advf_target_adapter(
            model,
            target_adapter_path=args.advf.target_adapter_path,
            dtype=model_dtype,
        )

    with torch.inference_mode():
        generate_raw_samples(
            model,
            tokenizer,
            raw_dataset,
            args.gen.gen_post_train_max_samples,
            batch_size=args.gen.gen_batch_size,
            max_new_tokens=args.gen.generate_max_new_tokens,
            save_path=os.path.join(results_dir, "fine_tuned"),
        )


if __name__ == "__main__":
    main()
