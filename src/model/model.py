from dataclasses import dataclass

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig


@dataclass
class ModelCategoryClass:
    LLAMA3: str = "llama-3"
    QWEN2_5: str = "qwen2.5"
    GEMMA: str = "gemma"


ModelCategories = ModelCategoryClass()


def init_model(
    model_name_or_path,
    quantization_mode=None,
):
    bnb_config = None
    model_dtype = None
    if quantization_mode == "4bit":
        print("Quantizing model to 4-bit")
        model_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )
    elif quantization_mode == "8bit":
        print("Quantizing model to 8-bit")
        model_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=model_dtype,
        device_map="auto",
        quantization_config=bnb_config,
        # attention_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    if model_dtype is None:
        model_dtype = model.dtype

    model_cat = next(
        (
            cat
            for cat in ModelCategories.__dict__.values()
            if isinstance(cat, str) and cat in model_name_or_path.lower()
        ),
        None,
    )
    if model_cat is None:
        raise ValueError(
            f"Model category not recognized for {model_name_or_path}"
        )

    model.config.model_name = model_name_or_path.split("/")[-1]
    model.config.model_path = model_name_or_path
    model.config.model_category = model_cat

    return model, model_dtype


def init_tokenizer(model_name_or_path, model):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        tokenizer.pad_token = pad_token
        assert (
            tokenizer.pad_token_id == pad_token_id
        ), f"Pad token ID mismatch: {tokenizer.pad_token_id} != {pad_token_id}"
        print(
            f"Setting pad token to {tokenizer.pad_token} ({tokenizer.pad_token_id})"
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    print(
        f"Tokenizer padding token: {tokenizer.pad_token} ({tokenizer.pad_token_id})"
    )
    print(f"Model padding token: {model.config.pad_token_id}")
    print(
        f"Tokenizer eos token: {tokenizer.eos_token} ({tokenizer.eos_token_id})"
    )
    print(f"Model eos token: {model.config.eos_token_id}")
    return tokenizer
