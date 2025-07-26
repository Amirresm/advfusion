from dataclasses import dataclass
import json
import os

import torch
import simple_parsing
from rich import print as rprint

from src.dataset.args import DatasetArgs
from src.dataset.utils import DatasetType
from src.generation.args import GenArgs
from src.generation.generation import generate_raw_samples
from src.model.args import ModelArgs
from src.model.model import init_model, init_tokenizer
from src.dataset.dataset import load_raw_dataset, preprocess_dataset
from src.model.utils import ModelType
from src.peft.args import PeftArgs
from src.peft.configs import get_peft_config
from src.peft.peft import load_peft, setup_for_peft
from src.train.args import TrainArgs
from src.train.trainer import get_trainer


@dataclass
class Args:
    model: ModelArgs
    peft: PeftArgs
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

    if args.peft.lib is not None and args.peft.peft_method is not None:
        if args.peft.preload_peft_from is not None:
            print(f"Loading PEFT model from {args.peft.preload_peft_from}")
            peft_config = get_peft_config(args.peft.peft_method, args.peft.lib)
            model = load_peft(
                model,
                args.model.model_type,
                peft_lib=args.peft.lib,
                peft_path=args.peft.preload_peft_from,
                dtype=model_dtype,
            )
        elif args.peft.peft_method is not None:
            peft_config = get_peft_config(args.peft.peft_method, args.peft.lib)
            model = setup_for_peft(
                model,
                args.model.model_type,
                args.peft.lib,
                config=peft_config,
                dtype=model_dtype,
            )
        else:
            print(
                "Warning: Peft library is specified but no PEFT config is provided."
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
            peft_lib=args.peft.lib,
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

        save_path = os.path.join(output_dir, "adapter")
        if args.peft.lib == "adp":
            model.save_adapter_setup(save_path, "adapter")
        else:
            model.save_pretrained(save_path)

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
