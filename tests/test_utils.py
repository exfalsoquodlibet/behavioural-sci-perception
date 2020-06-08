import pytest
import pandas as pd
import numpy as np
from typing import Callable
from pandas.testing import assert_frame_equal
from src.utils.make_pipeline import chain_2funcs, chain_functions
from src.make_data.preproc import replace_nan, to_lower_textfields, to_lower_cols


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


dummy_df = pd.DataFrame({
    'Role': ['xx', 'Zzz', 'Z zz', 'z', np.nan, np.nan, '', ''],
    'Role leVEL': ['YYY', '', np.nan, 'Ss S', np.nan, np.nan, 'J', 'q qq'],
    'role name': ['g', '', '', 't', 'o ooo', 'aa a a', 'u', 'd'],
    'Int_Col': [1, 2, 3, np.nan, 5, 6, 7, 8]
})

example_pipeline_1 = chain_functions(to_lower_cols, to_lower_textfields,
                                     replace_nan)


def test_chain_functions_integrated_1():
    assert isinstance(example_pipeline_1, Callable)
    assert_frame_equal(
        example_pipeline_1(dummy_df),
        replace_nan(to_lower_textfields(to_lower_cols(dummy_df))))
    assert_frame_equal(
        example_pipeline_1(dummy_df),
        pd.DataFrame({
            'role': ['xx', 'zzz', 'z zz', 'z', '', '', '', ''],
            'role level': ['yyy', '', '', 'ss s', '', '', 'j', 'q qq'],
            'role name': ['g', '', '', 't', 'o ooo', 'aa a a', 'u', 'd'],
            'int_col': [1, 2, 3, np.nan, 5, 6, 7, 8]
        }))


example_pipeline_2 = chain_functions(
    to_lower_cols, to_lower_textfields,
    lambda x: replace_nan(x, only_these_cols='role'))


def test_chain_functions_integrated_2():
    assert_frame_equal(
        example_pipeline_2(dummy_df),
        pd.DataFrame({
            'role': ['xx', 'zzz', 'z zz', 'z', '', '', '', ''],
            'role level':
            ['yyy', '', np.nan, 'ss s', np.nan, np.nan, 'j', 'q qq'],
            'role name': ['g', '', '', 't', 'o ooo', 'aa a a', 'u', 'd'],
            'int_col': [1, 2, 3, np.nan, 5, 6, 7, 8]
        }))
