# AdvFusion

This repository contains the source code for training and inference of adapters
and adversarial fusion.

# Getting Started

Use `uv sync` to create an environment and install the dependencies.

Alternatively, you can use `python -m venv .venv` to create a virtual
environment and use the dependencies in `pyproject.toml`.

## Training individual adapters

To train individual adapters, use the `scripts/train.py` script. With this
script, you can train and infer per-language adapters with Peft or AdapterHub
adapters. You can also fully fine-tune models with this script. Run with `-h`
for config and see `examples/train.sh` for usage example.

## Training adversarial fusion

To train adversarial fusion, use the `scripts/train_advf.py` script. With this
script, you can train and infer adversarial fusion with AdapterHub adapters.
Note that you need pre-trained language adapters to use this script. Run with
`-h` for config and see `examples/train_advf.sh` for usage example.

# Notes

## Supported Models

Currently, adversarial fusion is only supported for the following model
families:

- `Llama-2`
- `Llama-3`
- `Qwen-2.5`
- `Gemma-2`
- `DeepSeek-Coder`

Model type is detected from the model name or path. If your model's name or path
does not reflect the model family, you can use the `--model_type` argument to
force the model type. See `src/model/utils.py` for model types.

## Supported Datasets

Supporting new datasets is a little more involved. You need to add your dataset
type to `src/dataset/utils.py` and implement a custom row processor in
`./src/dataset/custom_processors.py`. The row processor takes a row and returns
an 'input' and 'target' for the model.
