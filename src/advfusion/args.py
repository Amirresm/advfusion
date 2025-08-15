from dataclasses import dataclass

from simple_parsing import field


@dataclass
class AdvfArgs:
    """AdvFusion arguments for loading and using adapters."""

    adapter_path_list: list[str] = field(
        required=True
    )  # List of adapter paths to load, separated by commas.
    target_adapter_path: str = field(
        required=True
    )  # Path to the target adapter to be used with AdvFusion.
    preload_advf_from: str | None = (
        None  # Path to the preloaded AdvFusion model, if any.
    )
