from dataclasses import dataclass


@dataclass
class GenArgs:
    do_generate: bool = True

    gen_batch_size: int = 4
    generate_max_new_tokens: int = 512

    gen_pre_train_max_samples: int | None = 0
    gen_post_train_max_samples: int | None = None
