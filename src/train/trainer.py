from typing import Literal
import adapters
from adapters.trainer import TrainingArguments
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
    logging_steps=100,
    eval_steps=500,
):
    data_collator = DefaultDataCollator()

    # callbacks = []

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        lr_scheduler_type="constant",
        optim="paged_adamw_32bit",
        warmup_ratio=0.03,
        learning_rate=1e-4,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_accumulation_steps=50,
        save_steps=eval_steps,
        save_total_limit=2,
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=True,
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
