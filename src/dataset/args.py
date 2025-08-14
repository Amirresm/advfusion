from dataclasses import dataclass, field

from src.dataset.utils import DatasetType


@dataclass
class DatasetArgs:
    """Dataset arguments."""

    dataset_name_or_path: str = field(
        metadata={
            "help": "Path to the dataset or dataset name.",
        }
    )
    dataset_type: DatasetType | None = field(
        default=None,
        metadata={
            "help": (
                "Type of the dataset to use (e.g., 'codesearchnet', 'codegeneration'). "
                "Leave empty to auto-detect."
            )
        },
    )
    train_file: str | None = field(
        default=None,
        metadata={
            "help": (
                "File name for the training dataset. If not provided, will use 'dataset_name_or_path' as the training dataset."
            ),
        },
    )
    validation_file: str | float | None = field(
        default=None,
        metadata={
            "help": (
                "File name for the validation dataset. Can be 0 < x < 1 to split the training dataset."
            ),
        },
    )
    test_file: str | float | None = field(
        default=None,
        metadata={
            "help": (
                "File name for the test dataset. Can be 0 < x < 1 to split the training dataset."
            ),
        },
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of samples to use for training.",
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of samples to use for evaluation.",
        },
    )
    max_test_samples: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of samples to use for test.",
        },
    )
    chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Chunk size for the dataset. If not provided or zero, will not chunk the dataset."
            ),
        },
    )
    train_completions_only: bool = (
        False  # Only calculate loss on completions, not on inputs.
    )

    train_text_max_length: int = field(
        default=512,
        metadata={
            "help": (
                "Maximum length of the text input. If not provided, defaults to 512. Not used when chunk_size is set."
            ),
        },
    )
    train_target_max_length: int = field(
        default=512,
        metadata={
            "help": (
                "Maximum length of the target output. If not provided, defaults to 128. Not used when chunk_size is set."
            ),
        },
    )

    valid_text_max_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum length of the text input for validation. If not provided, defaults train_text_max_length."
            ),
        },
    )
    valid_target_max_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum length of the target output for validation. If not provided, defaults train_target_max_length."
            ),
        },
    )
    test_text_max_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum length of the text input for test. If not provided, defaults train_text_max_length."
            ),
        },
    )
    test_target_max_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum length of the target output for test. If not provided, defaults train_target_max_length."
            ),
        },
    )

    def __post_init__(self):
        try:
            self.validation_file = (
                float(self.validation_file) if self.validation_file else None
            )
        except:
            pass
        try:
            self.test_file = float(self.test_file) if self.test_file else None
        except:
            pass

        if isinstance(self.dataset_type, str):
            self.dataset_type = DatasetType.get_dataset_type(self.dataset_type)
        elif self.dataset_type is None:
            self.dataset_type = DatasetType.get_dataset_type(
                self.dataset_name_or_path
            )
