import adapters

from src.model.utils import ModelType

CodeGemmaInterface = adapters.AdapterModelInterface(
    adapter_methods=["bottleneck", "lora", "reft", "invertible"],
    model_embeddings="embed_tokens",
    model_layers="layers",
    layer_self_attn="self_attn",
    layer_cross_attn=None,
    attn_k_proj="k_proj",
    attn_q_proj="q_proj",
    attn_v_proj="v_proj",
    attn_o_proj="o_proj",
    layer_intermediate_proj="mlp.up_proj",
    layer_output_proj="mlp.down_proj",
    layer_pre_self_attn="input_layernorm",
    layer_pre_ffn="post_attention_layernorm",
    layer_ln_1=None,
    layer_ln_2=None,
)

Llama3Interface = adapters.AdapterModelInterface(
    adapter_methods=["bottleneck", "lora", "reft", "invertible"],
    model_embeddings="embed_tokens",
    model_layers="layers",
    layer_self_attn="self_attn",
    layer_cross_attn=None,
    attn_k_proj="k_proj",
    attn_q_proj="q_proj",
    attn_v_proj="v_proj",
    attn_o_proj="o_proj",
    layer_intermediate_proj="mlp.up_proj",
    layer_output_proj="mlp.down_proj",
    layer_pre_self_attn="input_layernorm",
    layer_pre_ffn="post_attention_layernorm",
    layer_ln_1=None,
    layer_ln_2=None,
)


def get_adapter_interface(model_category: ModelType):
    if model_category == ModelType.CODELLAMA:
        return CodeGemmaInterface
    elif model_category in [
        ModelType.LLAMA2,
        ModelType.LLAMA3,
        ModelType.CODELLAMA,
    ]:
        return Llama3Interface
    elif model_category == ModelType.DEEPSEEK:
        return Llama3Interface
    else:
        return None
