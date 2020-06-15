import yaml
from typing import Callable
from functools import reduce


def chain_2funcs(f: Callable, g: Callable) -> Callable:
    """
    Chains two functions.
    The output of the first function `f` must be an acceptable input for the second function `g`.

    Args:
        f:   first function to be applied
        g:   second function to be applied to the output of f().

    Returns:
        A function that applies g() to the output of f().
    """
    return lambda *args, **kwargs: g(f(*args, **kwargs))


def chain_functions(*f_args: Callable) -> Callable:
    """
    Combines an n-th number of functions together.
    Functions will be applied in order from left to right. Example, chain_functions(f, g) for g(f(x))

    Args:
        f_args:   functions to be chained.

    Returns:
        A function that applies the chained functions.
    """
    return reduce(chain_2funcs, f_args, lambda x: x)


def load_config_yaml(config_file: str) -> dict:
    """Load YAML configuration file at path config_file

    Args:
        config_file (str): path to the yaml file

    Returns:
        the configuration object as a dictionary

    """

    try:
        with open(config_file, "r") as inp:
            config = yaml.load(inp, Loader=yaml.FullLoader)
        print(config_file, "has been successfully loaded as a dict")
        return config
    except FileNotFoundError as e:
        raise type(e)(config_file, "not found!")
    except IOError as e:
        raise type(e)("Couldn't open", config_file)


if __name__ == "__main__":

    import os
    # Load configuration file
    CONFIG_FILE_NAME = "config_news_data.yaml"
    DIR_EXT = os.environ.get("DIR_EXT")
    CONFIG_FILE = os.path.join(DIR_EXT, CONFIG_FILE_NAME)

    print(CONFIG_FILE)

    CONFIG = load_config_yaml(CONFIG_FILE)
