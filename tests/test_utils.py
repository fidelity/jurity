# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

import numpy as np
import pandas as pd

from jurity.utils import Constants
from jurity.utils import InputShapeError
from jurity.utils import check_and_convert_list_types, split_array_based_on_membership_label
from jurity.utils import check_binary, check_inputs_validity, check_or_convert_numpy_array
from jurity.utils import check_elementwise_input_type, check_input_type, check_input_shape
from jurity.utils import performance_measures
from jurity.utils import convert_one_vs_rest

with warnings.catch_warnings(record=False):
    pass


class TestUtils(unittest.TestCase):

    def test_check_input_type_list(self):
        my_list = [1, 1, 2]
        check_input_type(my_list)

    def test_check_input_type_arr(self):
        my_arr = np.array([1, 2])
        check_input_type(my_arr)

    def test_check_input_type_df(self):
        my_series = pd.DataFrame.from_dict({'a': [1, 2, 2]})['a']
        check_input_type(my_series)

    def test_check_input_type_invalid(self):
        my_set = {1, 2}
        with self.assertRaises(TypeError):
            check_input_type(my_set)

    def test_check_input_shape_list(self):
        my_list = [1, 1, 2]
        check_input_shape(my_list)

    def test_check_input_shape_np_arr_valid(self):
        my_arr = np.array([1, 2])
        check_input_shape(my_arr)

    def test_check_input_shape_np_arr_invalid(self):
        error_numpy_arr = np.array([[1, 2], [1, 2]])
        with self.assertRaises(InputShapeError):
            check_input_shape(error_numpy_arr)

    def test_check_input_shape_df_valid(self):
        my_df = pd.DataFrame.from_dict({'a': [1, 2, 2]})
        check_input_shape(my_df['a'])

    def test_check_input_shape_df_invalid(self):
        my_df = pd.DataFrame.from_dict({'a': [1, 2, 2]})
        with self.assertRaises(InputShapeError):
            check_input_shape(my_df)

    def test_check_binary_valid(self):
        arr = np.array([1, 2, 1, 2])
        check_binary(arr)

    def test_check_binary_invalid(self):
        arr = np.array([1, 2, 1, 3])
        with self.assertRaises(ValueError):
            check_binary(arr)

    def test_check_inputs_validity_valid(self):
        labels = np.array([1, 1, 0])
        y_pred = np.array([1, 1, 0])
        is_member = np.array([0, 0, 1])

        check_inputs_validity(y_pred, is_member, False, labels)

    def test_check_inputs_validity_invalid(self):
        labels = np.array([1, 0, 2])
        y_pred = np.array([1, 1, 0])
        is_member = np.array([0, 0, 1])

        with self.assertRaises(ValueError):
            check_inputs_validity(y_pred, is_member, False, labels)

    def test_check_elementwise_input_type_valid(self):
        arr = [1, 1, 0]
        check_elementwise_input_type(arr)

    def test_check_elementwise_input_type_invalid(self):
        arr = [1, 1., True]
        with self.assertRaises(TypeError):
            check_elementwise_input_type(arr)

    def test_check_elementwise_input_type_unsupported(self):
        arr = [1, 1, 'a']
        with self.assertRaises(TypeError):
            check_elementwise_input_type(arr)

    def test_check_binary_list(self):
        check_binary([1, 2, 2])

    def test_check_binary_np(self):
        check_binary(np.array([1, 2, 2]))

    def test_check_binary_df(self):
        check_binary(pd.DataFrame.from_dict({'a': [1, 2, 2]})['a'])

    def test_check_binary_invalid_type(self):
        with self.assertWarns(UserWarning):
            with self.assertRaises(TypeError):
                check_binary({1, 2})

    def test_check_binary_more_than_two_values(self):
        with self.assertRaises(ValueError):
            check_binary([1, 2, 3])

    def test_check_inputs_validity_missing_all(self):
        with self.assertRaises(ValueError):
            check_inputs_validity(None, None, None)

    def test_check_inputs_validity_missing_predictions(self):
        labels = [0, 1, 1]
        is_member = pd.DataFrame.from_dict({'a': [1, 2, 2]})['a']

        with self.assertRaises(ValueError):
            check_inputs_validity(labels=labels, predictions=None, is_member=is_member)

    def test_check_inputs_validity_missing_is_member(self):
        labels = [0, 1, 1]
        predictions = [3, 4, 3]

        with self.assertRaises(ValueError):
            check_inputs_validity(labels=labels, predictions=predictions, is_member=None)

    def test_check_inputs_validity_missing_label(self):
        predictions = [3, 4, 3]
        is_member = pd.DataFrame.from_dict({'a': [1, 2, 2]})['a']
        check_inputs_validity(labels=None, predictions=predictions, is_member=is_member)

    def test_performance_measures_no_group(self):
        ground_truth = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 0, 1])

        # test without group membership
        res1 = performance_measures(ground_truth, y_pred, group_membership=False)
        assert np.isclose(res1[Constants.TPR], 0.6666666, atol=0.001)
        assert res1[Constants.FPR] == 0.5
        assert np.isclose(res1[Constants.FNR], 0.333, atol=0.001)
        assert np.isclose(res1[Constants.PPV], 0.6666666, atol=0.001)
        assert res1[Constants.NPV] == 0.5
        assert np.isclose(res1[Constants.FDR], 0.333, atol=0.001)
        assert res1[Constants.FOR] == 0.5
        assert np.isclose(res1[Constants.ACC], 0.6, atol=0.001)

    def test_performance_measures_group(self):
        ground_truth = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        group_idx = [0, 1, 2]

        # test with group membership
        res1 = performance_measures(ground_truth, y_pred, group_idx, group_membership=True)
        assert res1[Constants.TPR] == 1
        assert res1[Constants.TNR] == 0.5
        assert res1[Constants.FPR] == 0.5
        assert res1[Constants.FNR] == 0
        assert res1[Constants.PPV] == 0.5
        assert res1[Constants.NPV] == 1
        assert res1[Constants.FDR] == 0.5
        assert res1[Constants.FOR] == 0
        assert np.isclose(res1[Constants.ACC], 0.6666666, atol=0.001)

    def test_check_and_convert_list_conversion_valid(self):
        my_list = [1, 2, 3]
        assert type(check_and_convert_list_types(my_list)) == np.ndarray

    def test_split_array_based_on_membership_label(self):
        ground_truth = np.array([0, 0, 0, 1, 1, 1])
        is_member = np.array([1, 1, 1, 0, 0, 0])
        privileged_truth, unprivileged_truth, _, _ = split_array_based_on_membership_label(ground_truth, is_member, 1)
        assert all(privileged_truth) == 1
        assert all(unprivileged_truth) == 0

    def test_split_array_based_on_membership_label_index_only(self):
        ground_truth = np.array([0, 0, 0, 1, 1, 1])
        is_member = np.array([1, 1, 1, 0, 0, 0])
        index_only = True
        membership_label = 1
        privileged_idx, unprivileged_idx = split_array_based_on_membership_label(ground_truth, is_member,
                                                                                 membership_label, index_only)
        assert list(privileged_idx) == [3, 4, 5]
        assert list(unprivileged_idx) == [0, 1, 2]

    def test_convert_or_check_numpy_arr_valid(self):
        arr = np.array([0, 0, 0, 1, 1, 1])
        arr = check_or_convert_numpy_array(arr)
        self.assertEqual(type(arr), np.ndarray)

    def test_convert_or_check_numpy_arr_pandas(self):
        arr = pd.Series(data=[0, 0, 0, 1, 1, 1], name='Data')
        arr = check_or_convert_numpy_array(arr)
        self.assertEqual(type(arr), np.ndarray)

    def test_convert_or_check_numpy_arr_list(self):
        arr = [0, 0, 0, 1, 1, 1]
        arr = check_or_convert_numpy_array(arr)
        self.assertEqual(type(arr), np.ndarray)

    def test_convert_or_check_numpy_arr_invalid(self):
        arr = str([0, 0, 0, 1, 1, 2])
        with self.assertRaises(TypeError):
            arr = check_or_convert_numpy_array(arr)

    def test_one_vs_rest(self):
        label = 'a'
        preds = ['a', 'b', 'c']
        converted = convert_one_vs_rest(label, preds)
        assert converted == [1, 0, 0]
