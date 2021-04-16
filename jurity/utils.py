# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import List, NamedTuple, NoReturn, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .hash_utils import lru_cache_df


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    default_seed = 1
    float_null = np.float64(0.0)
    TPR = "TPR"
    TNR = "TNR"
    FPR = "FPR"
    FNR = "FNR"
    PPV = "PPV"
    NPV = "NPV"
    FDR = "FDR"
    FOR = "FOR"
    ACC = "ACC"

    user_id = 'user_id'
    item_id = 'item_id'


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


def check_and_convert_list_types(arr: Union[List, np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Checks if input is a list and converts it to numpy array if so.
    """
    return np.array(arr) if isinstance(arr, list) else arr


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


def check_input_type(input_: Union[List, np.ndarray, pd.Series]) -> NoReturn:
    """
    Checks that input is a list, numpy array or pandas series, otherwise raises the given exception.
    """
    check_true(type(input_) == np.ndarray or
               type(input_) == list or
               type(input_) == pd.Series,
               TypeError("Incorrect input type."))


def check_input_shape(input_: Union[List, np.ndarray, pd.Series]) -> NoReturn:
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
            "Allowed data types (pandas df, list or numpy array) not passed in, cannot check for binary inputs.")
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


def check_elementwise_input_type(input_: Union[List, np.ndarray, pd.Series], binary_only: bool = True) -> NoReturn:
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

    if not binary_only:  # allow string as a class for multi-class
        check_str = all([isinstance(i, str) for i in input_]) is True
        check_list = all([isinstance(i, list) for i in input_]) is True
        checks.append(check_str)
        checks.append(check_list)
    all_checks = any(checks)

    check_true(all_checks, TypeError("Non uniform/unsupported data types"))


def check_inputs_validity(predictions: Union[List, np.ndarray, pd.Series],
                          is_member: Union[List, np.ndarray, pd.Series],
                          optional_labels: bool = True,
                          labels: Union[List, np.ndarray, pd.Series] = None,
                          binary_only: bool = True) -> NoReturn:
    """Checks that all given inputs are valid.

    Parameters
    ---------
    predictions: Union[List, np.ndarray, pd.Series]
        Predicted values.
    is_member: Union[List, np.ndarray, pd.Series]
        Group membership.
    optional_labels: bool
        True if labels are optional, False otherwise
    labels: Union[List, np.ndarray, pd.Series]
        Ground truth labels.
    binary_only: bool
        True if binary only, False otherwise

    Returns
    ---------
    None.
    """

    check_true(predictions is not None and is_member is not None,
               ValueError("You need to specify model predictions and is_member attribute"))

    # Check input types are in allowed types
    check_input_type(predictions)
    check_input_type(is_member)

    # Check input shapes are 1D
    check_input_shape(predictions)
    check_input_shape(is_member)

    # Check input content - only binary allowed
    if binary_only:
        check_binary(predictions)
        check_binary(is_member)

    check_elementwise_input_type(predictions, binary_only)
    check_elementwise_input_type(is_member, binary_only)

    # Check that our arrays are all the same length
    if not optional_labels:
        check_true(labels is not None,
                   ValueError("Labels are not optional for this metric, they need to be " "specified."))

        check_input_type(labels)
        check_input_shape(labels)
        check_binary(labels)
        check_elementwise_input_type(labels)

        check_true(len(labels) == len(predictions) == len(is_member),
                   InputShapeError("",
                                   f"Shapes of inputs do not match. "
                                   f"you supplied lengths of labels: "
                                   f"{len(labels)}, predictions: {len(predictions)}"
                                   f", is_member: {len(is_member)}."))
    else:
        check_true(len(predictions) == len(is_member),
                   InputShapeError("",
                                   f"Shapes of inputs do not match. "
                                   f"You supplied array lengths "
                                   f"predictions: {len(predictions)}, is_member: {len(is_member)}."))


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
    constants = Constants()

    return {constants.TPR: tp / p,
            constants.TNR: tn / n,
            constants.FPR: fp / n,
            constants.FNR: fn / p,
            constants.PPV: tp / (tp + fp) if (tp + fp) > 0.0 else Constants.float_null,
            constants.NPV: tn / (tn + fn) if (tn + fn) > 0.0 else Constants.float_null,
            constants.FDR: fp / (fp + tp) if (fp + tp) > 0.0 else Constants.float_null,
            constants.FOR: fn / (fn + tn) if (fn + tn) > 0.0 else Constants.float_null,
            constants.ACC: (tp + tn) / (p + n) if (p + n) > 0.0 else Constants.float_null}


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


def unique_multiclass_multilabel(input_: Union[List, List[List], np.ndarray, pd.Series]) -> np.ndarray:
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
