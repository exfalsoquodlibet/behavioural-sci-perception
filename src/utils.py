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
