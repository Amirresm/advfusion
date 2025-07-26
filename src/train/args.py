from dataclasses import dataclass


@dataclass
class TrainArgs:
    do_train: bool = False
    do_eval: bool = False

    train_batch_size: int = 4
    eval_batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03

    logging_steps: int = 100
    eval_steps: int = 200
