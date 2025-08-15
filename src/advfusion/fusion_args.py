from dataclasses import dataclass

from simple_parsing import field


@dataclass
class FusionArgs:
    """Fusion arguments for loading and using adapters."""

    adapter_path_list: list[str] = field(
        required=True
    )  # List of adapter paths to load, separated by commas.
    preload_fusion_from: str | None = (
        None  # Path to the preloaded Fusion model, if any.
    )
