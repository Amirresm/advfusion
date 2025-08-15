from typing import Literal
import adapters
from adapters.trainer import TrainingArguments
import torch
from transformers.data.data_collator import (
    DefaultDataCollator,
)
from transformers.trainer import Trainer


def get_trainer(
    model,
    peft_lib: Literal["adp", "peft"] | None,
    train_dataset,
    eval_dataset,
    output_dir,
    train_batch_size,
    eval_batch_size,
    epochs,
    learning_rate,
    warmup_ratio,
    logging_steps=0.05,
    eval_steps=0.1,
    eval_accumulation_steps=50,
    gradient_accumulation_steps=1,
):
    data_collator = DefaultDataCollator()

    # callbacks = []

    bf16 = model.config.torch_dtype == torch.bfloat16
    print(f"Using bf16 for training: {bf16}")

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        lr_scheduler_type="constant",
        optim="paged_adamw_32bit",
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=bf16,
        label_names=["labels"],
    )

    TrainerClass = adapters.AdapterTrainer if peft_lib == "adp" else Trainer
    trainer = TrainerClass(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # callbacks=callbacks,
    )

    return trainer
