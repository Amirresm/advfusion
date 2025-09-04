import json
import time
import torch

from rich import print


class ResourceLogger:
    def __init__(self, save_path):
        self.path = save_path
        self.usage = {}
        self.timestamp = None

    def clear_cuda(self, hard=False):
        torch.cuda.reset_peak_memory_stats()

        if hard:
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        self.timestamp = time.monotonic()

    def record(self, name, current=False):
        if name not in self.usage:
            self.usage[name] = {}

        if self.timestamp is None:
            self.timestamp = time.monotonic()
        elapsed = time.monotonic() - self.timestamp

        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        free_mem, total_mem = torch.cuda.mem_get_info()

        if current:
            main = allocated
        else:
            main = max_allocated

        self.usage[name] = {
            "main": main,
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated,
            "max_reserved": max_reserved,
            "elapsed": elapsed,
            "free_mem": free_mem,
            "total_mem": total_mem,
        }

        if "baseline" in self.usage and name != "baseline":
            baseline = self.usage["baseline"]

            baseline_main = main - baseline["main"]
            baseline_allocated = allocated - baseline["allocated"]
            baseline_reserved = reserved - baseline["reserved"]
            baseline_max_allocated = max_allocated - baseline["max_allocated"]
            baseline_max_reserved = max_reserved - baseline["max_reserved"]

            self.usage[name].update(
                {
                    "baseline_delta_main": baseline_main,
                    "baseline_delta_allocated": baseline_allocated,
                    "baseline_delta_reserved": baseline_reserved,
                    "baseline_delta_max_allocated": baseline_max_allocated,
                    "baseline_delta_max_reserved": baseline_max_reserved,
                }
            )

        self.usage[name]["name"] = name
        self.print(name)
        self.save()

    def record_baseline(self):
        self.record("baseline", current=True)

    def print(self, name=None):
        if name is not None:
            usage = self.usage[name]
            print(f"Resource usage for '{name}':")
            print(f"  * Main: {usage['main'] / 1e9:.3f} GB")
            print(f"  Allocated: {usage['allocated'] / 1e9:.3f} GB")
            print(f"  Reserved: {usage['reserved'] / 1e9:.3f} GB")
            print(f"  Max Allocated: {usage['max_allocated'] / 1e9:.3f} GB")
            print(f"  Max Reserved: {usage['max_reserved'] / 1e9:.3f} GB")
            print(
                f"  Elapsed Time: {usage['elapsed']:.3f} seconds ({usage['elapsed'] / 60:.2f} minutes)"
            )
            print(f"  Free Memory: {usage['free_mem'] / 1e9:.3f} GB")
            print(f"  Total Memory: {usage['total_mem'] / 1e9:.3f} GB")
            if "baseline_delta_allocated" in usage:
                print("  Delta from Baseline:")
                print(
                    f"    * Baseline Main: {usage['baseline_delta_main'] / 1e9:.3f} GB"
                )
                print(
                    f"    Baseline Allocated: {usage['baseline_delta_allocated'] / 1e9:.3f} GB"
                )
                print(
                    f"    Baseline Reserved: {usage['baseline_delta_reserved'] / 1e9:.3f} GB"
                )
                print(
                    f"    Baseline Max Allocated: {usage['baseline_delta_max_allocated'] / 1e9:.3f} GB"
                )
                print(
                    f"    Baseline Max Reserved: {usage['baseline_delta_max_reserved'] / 1e9:.3f} GB"
                )
            print()
        else:
            for key, value in self.usage.items():
                print(f"Resource usage for '{key}':")
                print(f"  * Main: {value['main'] / 1e9:.3f} GB")
                print(f"  Allocated: {value['allocated'] / 1e9:.3f} GB")
                print(f"  Reserved: {value['reserved'] / 1e9:.3f} GB")
                print(f"  Max Allocated: {value['max_allocated'] / 1e9:.3f} GB")
                print(f"  Max Reserved: {value['max_reserved'] / 1e9:.3f} GB")
                print(
                    f"  Elapsed Time: {value['elapsed']:.3f} seconds ({value['elapsed'] / 60:.2f} minutes)"
                )
                print(f"  Free Memory: {value['free_mem'] / 1e9:.3f} GB")
                print(f"  Total Memory: {value['total_mem'] / 1e9:.3f} GB")
                if "baseline_delta_allocated" in value:
                    print("  Delta from Baseline:")
                    print(
                        f"    * Baseline Main: {value['baseline_delta_main'] / 1e9:.3f} GB"
                    )
                    print(
                        f"    Baseline Allocated: {value['baseline_delta_allocated'] / 1e9:.3f} GB"
                    )
                    print(
                        f"    Baseline Reserved: {value['baseline_delta_reserved'] / 1e9:.3f} GB"
                    )
                    print(
                        f"    Baseline Max Allocated: {value['baseline_delta_max_allocated'] / 1e9:.3f} GB"
                    )
                    print(
                        f"    Baseline Max Reserved: {value['baseline_delta_max_reserved'] / 1e9:.3f} GB"
                    )
                print()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.usage, f, indent=4)
