from dataclasses import dataclass, field

from src.model.utils import ModelType


@dataclass
class ModelArgs:
    """Arguments for the model configuration."""
    model_name_or_path: str = field(
        metadata={
            "args": ["--model_name_or_path"],
            "help": "Path to the pre-trained model or model name.",
        }
    )
    model_type: ModelType | None = field(
        metadata={
            "type": str | None,
            "help": (
                "Type of the model to use (e.g., 'llama-3', 'codellama'). Leave empty to auto-detect."
            ),
        },
    )
    q: str | None = field(
        metadata={
            "args": ["--quantization_mode"],
            "help": (
                "Quantization mode to use for the model. Options are '4bit' or '8bit'."
            ),
        },
    )

    def __post_init__(self):
        if isinstance(self.model_type, str):
            self.model_type = ModelType.get_model_type(self.model_type)
        elif self.model_type is None:
            self.model_type = ModelType.get_model_type(self.model_name_or_path)
