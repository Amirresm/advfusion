from dataclasses import dataclass

from src.dataset.utils import DatasetType


@dataclass
class GenArgs:
    """Generation arguments."""

    do_generate: bool = True

    gen_batch_size: int = 4
    generate_max_new_tokens: int = 512

    gen_pre_train_max_samples: int | None = 0
    gen_post_train_max_samples: int | None = None

    gen_do_sample: bool = False
    gen_temperature: float | None = None
    gen_top_k: int | None = None
    gen_top_p: float | None = None
    gen_n_per_sample: int = 1

    benchmark_dataset_name_or_path: str | None = None
    benchmark_dataset_type: DatasetType | None = None

    benchmark_do_sample: bool = False
    benchmark_temperature: float | None = None
    benchmark_top_k: int | None = None
    benchmark_top_p: float | None = None
    benchmark_n_per_sample: int = 1

    def __post_init__(self):
        if isinstance(self.benchmark_dataset_type, str):
            self.benchmark_dataset_type = DatasetType.get_dataset_type(
                self.benchmark_dataset_type
            )
        elif (
            self.benchmark_dataset_type is None
            and self.benchmark_dataset_name_or_path
        ):
            self.benchmark_dataset_type = DatasetType.get_dataset_type(
                self.benchmark_dataset_name_or_path
            )
