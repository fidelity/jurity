# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, NoReturn, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix

from jurity.constants import Constants
from .utils_hash import lru_cache_df


class Error(Exception):
    """
    Base class for exceptions in this module.
    """
    pass


class InputShapeError(Error):
    """
    Exception raised for error from input shape.
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class NotFittedError(Error):
    """
    Exception raised for error from unfitted object.
    """

    def __init__(self, message):
        self.message = message


class WeightTooLarge(Error):
    """
    Exception for bootstrap calculations when input data is too small for weighted least squares
    """

    def __init__(self, message):
        self.message = message


def check_false(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is false, otherwise raises the given exception.
    """
    if expression:
        raise exception


def check_true(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is true, otherwise raises the given exception.
    """
    if not expression:
        raise exception


def check_and_convert_list_types(arr: Union[List, np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Checks if input is a list and converts it to numpy array if so.
    """
    return np.array(arr) if isinstance(arr, list) else arr


def check_or_convert_numpy_array(arr: Union[List, np.ndarray, pd.Series], error_message=""):
    """
    Convert input list, array, series to numpy array.
    """
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, pd.Series):
        return arr.to_numpy()
    elif isinstance(arr, list):
        return np.array(arr)
    else:
        raise TypeError(error_message)


def split_array_based_on_membership_label(arr_to_split: Union[List, np.ndarray, pd.Series],
                                          is_member: Union[List, np.ndarray, pd.Series],
                                          membership_label: Union[str, float, int],
                                          return_index_only: bool = False):
    """Splits input array based on group membership.

    Parameters
    ---------
    arr_to_split: Union[List, np.ndarray, pd.Series]
        Input to split based on group membership.
    is_member: Union[List, np.ndarray, pd.Series]
        Group membership for each input.
    membership_label: Union[str, float, int]
        Value used to indicate group membership.
    return_index_only:
        Flag indicating whether to only return indices for each group.

    Returns
    ---------
    Indices of non-members and members and as well as the split array.
    """

    # Identify members and non-members based on membership label
    nonmember_indices = np.where(is_member != membership_label)[0]
    member_indices = np.where(is_member == membership_label)[0]

    if return_index_only:
        return nonmember_indices, member_indices

    # Get members and non-member values based on membership label
    nonmember_values = arr_to_split[nonmember_indices]
    member_values = arr_to_split[member_indices]

    return nonmember_values, member_values, nonmember_indices, member_indices


def check_input_type(input_: Union[List, np.ndarray, pd.Series]) -> NoReturn:
    """
    Checks that input is a list, numpy array or pandas series, otherwise raises the given exception.
    """
    check_true(type(input_) == np.ndarray or
               type(input_) == list or
               type(input_) == pd.Series,
               TypeError("Incorrect input type."))


def check_input_1d(input_: Union[List, np.ndarray, pd.Series]) -> NoReturn:
    """
    Checks that input shape is 1d.
    """
    if type(input_) != list and input_ is not None:
        check_true(len(input_.shape) == 1,
                   InputShapeError(input_.shape, "Input shape needs to be 1D"))


def check_binary(input_: Union[List, np.ndarray, pd.Series]) -> NoReturn:
    """
    Check that input array contains only two distinct values.
    """
    if input_ is None:
        pass
    elif isinstance(input_, pd.Series):
        check_true(len(input_.unique().tolist()) <= 2,
                   ValueError(f"Only binary content allowed, you supplied: "
                              f"{input_.unique().tolist()}."))
    elif isinstance(input_, list) or isinstance(input_, np.ndarray):
        check_true(len(np.unique(input_)) <= 2,
                   ValueError(f"Only binary content allowed, you supplied: {np.unique(input_)}."))
    else:
        warnings.warn(
            "Allowed data types (list, numpy array, or pandas series) not passed in, cannot check for binary inputs.")
        check_input_type(input_)


def check_binary_values(input_: Union[List, np.ndarray, pd.Series]):
    unique_values = np.unique(input_)
    if len(unique_values) > 2:
        raise ValueError("Input must be 0 or 1.")
    for v in unique_values:
        if v not in [0, 1]:
            raise ValueError("Input must be 0 or 1.")


def check_likelihood_values(input_: Union[List, np.ndarray, pd.Series]):
    if np.min(input_) < 0 or np.max(input_) > 1:
        raise ValueError("Likelihoods must be between 0 and 1 (inclusive).")


def check_elementwise_input_type(input_: Union[List, np.ndarray, pd.Series], is_multi_class: bool = False) -> NoReturn:
    """
    Check that values in input array are all from same, allowable data type.
    """
    check_ints = all([isinstance(i, int) for i in input_]) is True
    check_floats = all([isinstance(i, float) for i in input_]) is True
    check_booleans = all([isinstance(i, bool) for i in input_]) is True
    check_numpy_ints32 = all([isinstance(i, np.int32) for i in input_]) is True
    check_numpy_ints64 = all([isinstance(i, np.int64) for i in input_]) is True
    check_numpy_booleans = all([isinstance(i, bool) for i in input_]) is True
    check_numpy_floats32 = all([isinstance(i, np.float32) for i in input_]) is True
    check_numpy_floats64 = all([isinstance(i, np.float64) for i in input_]) is True

    check_uint8 = all([isinstance(i, np.uint8) for i in input_]) is True
    check_uint16 = all([isinstance(i, np.uint16) for i in input_]) is True
    check_uint32 = all([isinstance(i, np.uint32) for i in input_]) is True

    checks = [check_ints,
              check_floats,
              check_booleans,
              check_numpy_ints32,
              check_numpy_ints64,
              check_numpy_booleans,
              check_numpy_floats32,
              check_numpy_floats64,
              check_uint8,
              check_uint16,
              check_uint32]

    if is_multi_class:  # allow string as a class for multi-class
        check_str = all([isinstance(i, str) for i in input_]) is True
        check_list = all([isinstance(i, list) for i in input_]) is True
        checks.append(check_str)
        checks.append(check_list)
    all_checks = any(checks)

    check_true(all_checks, TypeError("Non uniform/unsupported data types"))


def is_one_dimensional(array):
    # If pd series, or 1d np array, or 1d list, than it is one dimensional
    if isinstance(array, pd.Series) and array.dtype != 'object':
        return True
    elif type(array) == list:
        #print(array[0])
        #print(type(array[0]))
        #print(isinstance(array[0], np.ndarray))
        if type(array[0]) != list and (not isinstance(array[0], np.ndarray)):
            return True
        else:
            return False
    elif isinstance(array, np.ndarray):
        if not type(array[0]) == list and array.ndim == 1:
            return True
        else:
            return False
    else:
        return False


def performance_measures(ground_truth: np.ndarray,
                         predictions: np.ndarray,
                         group_idx: Optional[Union[np.ndarray, List]] = None,
                         group_membership: bool = False) -> dict:
    """Compute various performance measures, optionally conditioned on protected attribute.
    Assume that positive label is encoded as 1 and negative label as 0.

    Parameters
    ---------
    ground_truth: np.ndarray
        Ground truth labels (1/0).
    predictions: np.ndarray
        Predicted values.
    group_idx: Union[np.ndarray, List]
        Indices of the group to consider. Optional.
    group_membership: bool
        Restrict performance measures to members of a certain group.
        If None, the whole population is used.
        Default value is False.

    Returns
    ---------
    Dictionary with performance measure identifiers as keys and their corresponding values.
    """

    if group_membership:
        ground_truth = ground_truth[group_idx]
        predictions = predictions[group_idx]

    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()

    p = np.sum(ground_truth == 1)
    n = np.sum(ground_truth == 0)

    return {Constants.TPR: tp / p,
            Constants.TNR: tn / n,
            Constants.FPR: fp / n,
            Constants.FNR: fn / p,
            Constants.PPV: tp / (tp + fp) if (tp + fp) > 0.0 else Constants.float_null,
            Constants.NPV: tn / (tn + fn) if (tn + fn) > 0.0 else Constants.float_null,
            Constants.FDR: fp / (fp + tp) if (fp + tp) > 0.0 else Constants.float_null,
            Constants.FOR: fn / (fn + tn) if (fn + tn) > 0.0 else Constants.float_null,
            Constants.ACC: (tp + tn) / (p + n) if (p + n) > 0.0 else Constants.float_null}


def check_inputs(predictions: Union[List, np.ndarray, pd.Series],
                 memberships: Union[List, np.ndarray, pd.Series],
                 membership_labels: Union[int, float, str, List[int]],
                 must_have_labels: bool = False,
                 labels: Union[List, np.ndarray, pd.Series] = None,
                 is_multi_class: bool = False) -> NoReturn:
    """
    Checks that all given inputs are valid.

    Parameters
    ---------
    predictions: Union[List, np.ndarray, pd.Series]
        Predicted values.
    memberships: Union[List, np.ndarray, pd.Series]
        Group membership.
    membership_labels: Union[int, float, str, List[int]]
        Labels indicating membership.
    must_have_labels: bool
        True must have labels, False otherwise
    labels: Union[List, np.ndarray, pd.Series]
        Ground truth labels.
    is_multi_class: bool
        True if multi-class, False if binary
    Returns
    ---------
    None.
    """

    # Need predictions and memberships
    check_true(predictions is not None and memberships is not None,
               ValueError("You need to specify model predictions and is_member attribute"))

    # Check input types are in allowed types
    check_input_type(predictions)
    check_input_type(memberships)
    check_true(type(membership_labels) in (int, str, float),
               TypeError("Membership label type should be a single int/str primitive"))

    # Check input shapes are 1D
    check_input_1d(predictions)
    check_input_1d(memberships)

    # Check input content - only binary allowed
    # TODO not sure if this is used/called as False, ever
    if not is_multi_class:
        check_binary(predictions)
        check_binary(memberships)

    check_elementwise_input_type(predictions, is_multi_class)
    check_elementwise_input_type(memberships, is_multi_class)

    # Check that our arrays are all the same length
    if must_have_labels:
        check_true(labels is not None, ValueError("Metric must have labels"))

        check_input_type(labels)
        check_input_1d(labels)
        check_binary(labels)
        check_elementwise_input_type(labels)

        check_true(len(labels) == len(predictions) == len(memberships),
                   InputShapeError("",
                                   f"Shapes of inputs do not match. "
                                   f"you supplied lengths of labels: "
                                   f"{len(labels)}, predictions: {len(predictions)}"
                                   f", is_member: {len(memberships)}."))
    else:
        check_true(len(predictions) == len(memberships),
                   InputShapeError("",
                                   f"Shapes of inputs do not match. "
                                   f"You supplied array lengths "
                                   f"predictions: {len(predictions)}, is_member: {len(memberships)}."))


@lru_cache_df(maxsize=5)
def get_sorted_clicks(results, user_id_column, click_column, k):
    sorted_clicks = results.sort_values(click_column, ascending=False).groupby(user_id_column).head(k)
    return sorted_clicks


def convert_one_vs_rest(positive_label, predictions):
    """
    Helper function to binarize multi-class predictions to one vs. rest.

    :param positive_label: label to be specified as "1"
    :param predictions: list to be converted to binary predictions
    :return: binarized 0,1 list of predictions

    EXAMPLE:
    >>> predictions = ['a', 'b', 'c']
    >>> positive_label = 'a'
    >>> convert_one_vs_rest(positive_label, predictions)
    OUT: [1, 0, 0]
    """
    return list((np.array(predictions) == positive_label).astype(int))


def get_unique_values(input_: Union[List, List[List], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Method to transform input of various formats into a numpy array of unique elements.

    :param input_: (numpy arr, pandas dataframe series, list, list of lists)
    :return: np.ndarray

    EXAMPLE 1:
    >> arr = [1, 2, 2]
    >> unique_multiclass_multilabel(arr)
    OUT: [1, 2]

    EXAMPLE 2:
    >> arr = [1, [2], [1, 2]]
    >> unique_multiclass_multilabel(arr)
    OUT: [1, 2]

    """
    if input_ is None or len(input_) == 0:
        return np.array([])
    elif isinstance(input_, pd.Series):
        return input_.explode().unique()
    elif isinstance(input_, list) or isinstance(input_, np.ndarray):
        return np.unique(np.hstack(input_))
    else:
        raise TypeError(f"(numpy arr, pandas series, list, list of lists) supported. You supplied {type(input_)}")


def _get_integer_id_map(df: pd.DataFrame, row_id_column: str, col_id_column: str):
    """
    Create two mappings from original row and col ids used in the DataFrame to integer ids respectively.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame with (row_id, col_id) in each row.
    row_id_column: str
        Row_id column name.
    col_id_column: str
        Col_id column name.

    Return
    -------
    A mapping from original row_id to integer id, and a mapping from original col_id to integer id.
    """

    # Map each row_id to (0, n_rows)
    unique_row_ids = list(df[row_id_column].unique())
    row_id_map = dict(zip(unique_row_ids, range(len(unique_row_ids))))

    # Map each col_id to (0, n_cols)
    unique_col_ids = list(df[col_id_column].unique())
    col_id_map = dict(zip(unique_col_ids, range(len(unique_col_ids))))

    return row_id_map, col_id_map


def tocsr(df: pd.DataFrame, row_id_column: str, col_id_column: str):
    """
    Transform data frame with row_id and col_id columns to sparse csr matrix.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame with (row_id, col_id) in each row.
    row_id_column: str
        Row_id column name.
    col_id_column: str
        Col_id column name.

    Return
    -------
    Sparse matrix capturing the interactions between row_id and col_id in the given data frame.
    """

    row_id_map, col_id_map = _get_integer_id_map(df, row_id_column, col_id_column)

    # Update row_id, col_id values
    integer_row_ids = df[row_id_column].map(row_id_map).values
    integer_col_ids = df[col_id_column].map(col_id_map).values

    # Convert to csr matrix
    csr_matrix = sp.coo_matrix((np.ones(len(integer_row_ids)), (integer_row_ids, integer_col_ids))).tocsr()

    return csr_matrix


def sample_users(df: pd.DataFrame, user_id_column: str = Constants.user_id,
                 user_sample_size: Union[int, float, None] = 10000, seed: int = Constants.default_seed) -> pd.DataFrame:
    """
    Samples input data frame by selecting a random sample of users.

    Parameters
    ----------
    df: pd.DataFrame
        Data frame with a user_id_col column.
    user_id_column: str
        User id column name.
    user_sample_size: Union[int, float, None]
        When input is an integer, it defines the number of randomly sampled users. When input is float, it defines the
        proportion of users to randomly sample for evaluation. If it is None, all users are used for evaluation.
    seed : int, default=Constants.default_seed
        The seed used to create random state.

    Returns
    -------
    Sampled data frame
    """
    rng = np.random.default_rng(seed)
    users = df[user_id_column].unique()

    if isinstance(user_sample_size, float):
        check_true(0.0 < user_sample_size, ValueError("User_sample_size should be greater than 0.0", user_sample_size))
        if user_sample_size > 1.0:
            warnings.warn('User_sample_size can not be larger than total number of users. '
                          'User_sample_size is set to be 1.0 instead.')
            user_sample_size = 1.0

        users = rng.choice(users, size=int(len(users) * user_sample_size), replace=False)

    elif isinstance(user_sample_size, int):
        check_true(0 < user_sample_size, ValueError("User_sample_size should be greater than 0", user_sample_size))
        if user_sample_size > len(users):
            warnings.warn('User_sample_size can not be larger than total number of users. '
                          'User_sample_size is set to be the total number of users.')
            user_sample_size = len(users)

        users = rng.choice(users, size=user_sample_size, replace=False)

    return df[df[user_id_column].isin(users)]
