# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# From https://github.com/microsoft/recommenders/blob/acd9f39b3b9f8f696cdec4ff40189749426185f5/reco_utils/dataset/pandas_df_utils.py
# latest version: https://github.com/microsoft/recommenders/blob/master/reco_utils/dataset/pandas_df_utils.py

from functools import lru_cache, wraps
from typing import Optional

import pandas as pd


class PandasHash:
    """
    Wrapper class to allow pandas objects (DataFrames or Series) to be hashable
    """
    # https://github.com/microsoft/recommenders/blob/master/reco_utils/dataset/pandas_df_utils.py
    __slots__ = "pandas_object"

    def __init__(self, pandas_object):
        """Initialize class

        Args:
            pandas_object (pd.DataFrame|pd.Series): pandas object
        """

        if not isinstance(pandas_object, (pd.DataFrame, pd.Series)):
            raise TypeError("Can only wrap pandas DataFrame or Series objects")
        self.pandas_object = pandas_object

    def __eq__(self, other):
        """Overwrite equality comparison

        Args:
            other (pd.DataFrame|pd.Series): pandas object to compare
        Returns:
            bool: whether other object is the same as this one
        """

        return hash(self) == hash(other)

    def __hash__(self):
        """Overwrite hash operator for use with pandas objects
        Returns:
            int: hashed value of object
        """

        hashable = tuple(self.pandas_object.values.tobytes())
        if isinstance(self.pandas_object, pd.DataFrame):
            hashable += tuple(self.pandas_object.columns)
        else:
            hashable += tuple(self.pandas_object.name)
        return hash(hashable)


def lru_cache_df(maxsize: Optional[int], typed: bool = False):
    """Least-recently-used cache decorator for pandas Dataframes.

    Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls. It can
    save time when an expensive or I/O bound function is periodically called with the same arguments.
    Inspired in the `lru_cache function <https://docs.python.org/3/library/functools.html#functools.lru_cache>`_.
    Args:
        maxsize (int|None): max size of cache, if set to None cache is boundless
        typed (bool): arguments of different types are cached separately
    """

    def to_pandas_hash(val):
        """Return PandaHash object if input is a DataFrame otherwise return input unchanged"""
        return PandasHash(val) if isinstance(val, pd.DataFrame) else val

    def from_pandas_hash(val):
        """Extract DataFrame if input is PandaHash object otherwise return input unchanged"""
        return val.pandas_object if isinstance(val, PandasHash) else val

    def decorating_function(user_function):
        @wraps(user_function)
        def wrapper(*args, **kwargs):
            # convert DataFrames in args and kwargs to PandaHash objects
            args = tuple([to_pandas_hash(a) for a in args])
            kwargs = {k: to_pandas_hash(v) for k, v in kwargs.items()}
            return cached_wrapper(*args, **kwargs)

        @lru_cache(maxsize=maxsize, typed=typed)
        def cached_wrapper(*args, **kwargs):
            # get DataFrames from PandaHash objects in args and kwargs
            args = tuple([from_pandas_hash(a) for a in args])
            kwargs = {k: from_pandas_hash(v) for k, v in kwargs.items()}
            return user_function(*args, **kwargs)

        # retain lru_cache attributes
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorating_function
