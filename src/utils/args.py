from typing import TypeVar
import simple_parsing

T = TypeVar("T", bound="simple_parsing.Serializable")

def parse_args(cls: type[T]) -> T:
    # parser = simple_parsing.ArgumentParser()
    # parser.add_argument(
    #     "--config", type=str, default=None, help="Path to the config file."
    # )
    # args, _ = parser.parse_known_args()
    #
    # if args.config:
    #     args = cls.load_yaml(args.config)
    # else:
    #     args = simple_parsing.parse(cls)

    args = simple_parsing.parse(cls)
    return args
