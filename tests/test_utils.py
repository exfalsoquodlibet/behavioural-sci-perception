import pytest
from typing import Callable
from src.utils import chain_2funcs, chain_functions


def f1(n: int) -> str:
    return str(n)


def f2(s: str) -> str:
    return "Hello " + s + "!"


def f3(s: str, upper=False) -> str:
    output = "Hello " + s + "!"
    return output.upper() if upper else output


def f4(n: int, factor: int = 2) -> int:
    return n * factor


args_chain_2funcs_expected = [(f1, f2, [42], "Hello 42!"),
                              (f1, lambda x: f3(x, upper=True), [42],
                               "HELLO 42!"), (f4, f1, [14, 3], "42")]


@pytest.mark.parametrize("test_arg1, test_arg2, test_input, test_expected",
                         args_chain_2funcs_expected)
def test_chain_2funcs_expected(test_arg1, test_arg2, test_input,
                               test_expected):
    chained_funcs = chain_2funcs(test_arg1, test_arg2)
    assert isinstance(chained_funcs, Callable)
    assert chained_funcs(*test_input) == test_expected


def test_chain_2funcs_errors():
    chained_funcs42 = chain_2funcs(f4, f2)
    with pytest.raises(TypeError):
        chained_funcs42(14, 3)


args_chain_functions_expected = [
    ([f4, f1, f2], 42, f2(f1(f4(42)))),
    ([f4, f1, f2], 42, "Hello 84!"),
    ([lambda x: f4(x, factor=3), f1,
      lambda x: f3(x, upper=True)], 14, f3(f1(f4(14, 3)), upper=True)),
    ([lambda x: f4(x, factor=3), f1,
      lambda x: f3(x, upper=True)], 14, "HELLO 42!"),
]


@pytest.mark.parametrize("test_input_funcs, test_input, test_expected",
                         args_chain_functions_expected)
def test_chain_functions_expected(test_input_funcs, test_input, test_expected):
    chained_funcs = chain_functions(*test_input_funcs)
    assert chained_funcs(test_input) == test_expected
