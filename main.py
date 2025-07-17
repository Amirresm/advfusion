import adapters
import peft

from src.model.report import report_model
from src.generation.generation import generate_raw_samples
from src.model.model import init_model, init_tokenizer
from src.dataset.dataset import load_raw_dataset, preprocess_dataset
from src.peft.peft import setup_for_peft
from src.train.train import get_trainer


def main():
    model_name_or_path = "/home/amirreza/projects/ai/models/llm/llama-3.2-3B"
    quantization_mode = "4bit"

    model, model_dtype = init_model(
        model_name_or_path,
        quantization_mode=quantization_mode,
    )
    tokenizer = init_tokenizer(model_name_or_path, model)

    print(f"Model loaded with dtype: {model_dtype}")

    dataset_name_or_path = "~/projects/ai/data/CodeSearchNet/python"

    raw_dataset = load_raw_dataset(
        dataset_name_or_path,
        train_file="train.jsonl",
        test_file="test.jsonl",
        validation_file="valid.jsonl",
    )

    generate_raw_samples(model, tokenizer, raw_dataset["test"][:0])

    peft_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        # task_type=peft.TaskType.CAUSAL_LM,
        task_type=None,
    )
    peft_method = "peft"
    peft_config = adapters.SeqBnConfig()
    # peft_config = adapters.LoRAConfig(
    #     r=8,
    #     alpha=16,
    #     dropout=0.1,
    #     selfattn_lora=True,
    #     intermediate_lora=False,
    #     output_lora=False,
    #     attn_matrices=["q", "v"],
    # )
    peft_method = "adp"

    model = setup_for_peft(
        model, peft_method, config=peft_config, dtype=model_dtype
    )

    report_model(model)

    text_column = "code"
    target_column = "docstring"

    def join_text_target(text: str, target: str) -> str:
        return f"{text}\nExplanation:\n{target}"

    train_dataset = preprocess_dataset(
        raw_dataset,
        "train",
        tokenizer=tokenizer,
        text_column=text_column,
        target_column=target_column,
        join_text_target=join_text_target,
        max_sample_count=10000,
        # do_chunk_text=False,
        chunk_size=256,
    )
    eval_dataset = preprocess_dataset(
        raw_dataset,
        "validation",
        tokenizer=tokenizer,
        text_column=text_column,
        target_column=target_column,
        join_text_target=join_text_target,
        max_sample_count=50,
        # do_chunk_text=False,
        chunk_size=256,
    )

    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        peft_lib=peft_method,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    generate_raw_samples(model, tokenizer, raw_dataset["test"][:10])


if __name__ == "__main__":
    main()
