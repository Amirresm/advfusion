from enum import Enum


class ModelType(Enum):
    LLAMA3 = "llama-3"
    LLAMA2 = "llama-2"
    CODELLAMA = "codellama"
    QWEN2_5 = "qwen2.5"
    GEMMA = "gemma"
    DEEPSEEK = "deepseek"

    @classmethod
    def get_model_type(cls, model_name_or_path: str) -> "ModelType":
        model_name_or_path = model_name_or_path.lower()
        for category in cls:
            if category.value in model_name_or_path:
                return category
        raise ValueError(f"Model category not recognized for {model_name_or_path}")
