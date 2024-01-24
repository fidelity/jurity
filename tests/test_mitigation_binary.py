# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import gc
import io
import pickle
import unittest

import numpy as np
import pandas as pd

from jurity.fairness import BinaryFairnessMetrics
from jurity.mitigation import BinaryMitigation
from jurity.utils import Constants, InputShapeError, NotFittedError


class TestBinaryMitigation(unittest.TestCase):

    def test_repr(self):
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        representation = mitigation.__repr__()

        self.assertTrue('P2P group 0' in representation)

    def test_usage(self):

        # Data
        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        # Bias Mitigation
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        # Training: Learn mixing rates from labelled data
        mitigation.fit(labels, predictions, likelihoods, is_member)

        # Testing: Mitigate bias in predictions
        fair_predictions, fair_likelihoods = mitigation.transform(predictions, likelihoods, is_member)

    def test_usage_pickle(self):

        # Fit a mitigation object, serialize to pickle
        # Then unserialize pickle and transform input
        # It should give the same results from using without pickling

        df = self.generate_random_population(num_individuals=500,
                                             threshold=0.5)

        np.random.seed(Constants.default_seed)

        # Randomly split the data into two sets, one for computing the mixing rate and one evaluating the fairness
        order = np.random.permutation(len(df))
        train_indices = order[0::2]
        test_indices = order[1::2]
        train_data = df.iloc[train_indices].copy()
        test_data = df.iloc[test_indices].copy()

        # Mitigation object
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        # Train data
        train_labels = train_data.label.to_numpy()
        train_predictions = train_data.y_pred_binary.to_numpy()
        train_likelihoods = train_data.prediction.to_numpy()
        train_is_member = train_data.group.to_numpy()

        # Fit
        mitigation.fit(train_labels, train_predictions, train_likelihoods, train_is_member)

        # Pickle model object
        bytes_io = io.BytesIO()
        pickle.dump(mitigation, bytes_io, pickle.HIGHEST_PROTOCOL)
        del mitigation
        gc.collect()
        bytes_io.seek(0)
        mitigation = pickle.load(bytes_io)

        # Test data
        test_labels = test_data.label.to_numpy()
        test_predictions = test_data.y_pred_binary.to_numpy()
        test_likelihoods = test_data.prediction.to_numpy()
        test_is_member = test_data.group.to_numpy()

        fair_predictions, fair_likelihoods = mitigation.transform(test_predictions, test_likelihoods, test_is_member)

        fair_df = pd.DataFrame.from_dict({
            'label': test_labels,
            'group': test_is_member,
            'y_pred': fair_predictions.astype(int)})

        fairness_metrics = BinaryFairnessMetrics().get_all_scores(fair_df.label.tolist(),
                                                                  fair_df.y_pred.tolist(),
                                                                  fair_df.group.tolist()).reset_index()

        ao = fairness_metrics.loc[fairness_metrics.Metric == 'Average Odds', 'Value'].to_numpy().item()
        di = fairness_metrics.loc[fairness_metrics.Metric == 'Disparate Impact', 'Value'].to_numpy().item()
        eq = fairness_metrics.loc[fairness_metrics.Metric == 'Equal Opportunity', 'Value'].to_numpy().item()
        fnr = fairness_metrics.loc[fairness_metrics.Metric == 'FNR difference', 'Value'].to_numpy().item()
        gei = fairness_metrics.loc[fairness_metrics.Metric == 'Generalized Entropy Index', 'Value'].to_numpy().item()
        pe = fairness_metrics.loc[fairness_metrics.Metric == 'Predictive Equality', 'Value'].to_numpy().item()
        sp = fairness_metrics.loc[fairness_metrics.Metric == 'Statistical Parity', 'Value'].to_numpy().item()
        ti = fairness_metrics.loc[fairness_metrics.Metric == 'Theil Index', 'Value'].to_numpy().item()

        self.assertEqual(ao, 0.049)
        self.assertEqual(di, 1.072)
        self.assertEqual(eq, 0.060)
        self.assertEqual(fnr, -0.060)
        self.assertEqual(gei, 0.102)
        self.assertEqual(pe, 0.037)
        self.assertEqual(sp, 0.062)
        self.assertEqual(ti, 0.133)

    def test_labelwise_rates(self):

        # Randomly generated 500 data points with biased outcome
        df = self.generate_random_population(num_individuals=500,
                                             threshold=0.5)

        labels = df.label.to_numpy()
        predictions = df.y_pred_binary.to_numpy()

        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate = \
            mitigation._get_label_wise_rates(labels, predictions)

        self.assertEqual(true_positive_rate, 0.358)
        self.assertEqual(false_positive_rate, 0.224)
        self.assertEqual(true_negative_rate, 0.226)
        self.assertEqual(false_negative_rate, 0.192)

    def test_numerical_stability_mixing_rate_small(self):

        # Randomly generated 500 data points with biased outcome
        df = self.generate_random_population(num_individuals=500,
                                             threshold=0.5)

        # Data
        labels = df.label.to_numpy()
        predictions = df.y_pred_binary.to_numpy()
        likelihoods = df.prediction.to_numpy()
        is_member = df.group.to_numpy()

        # Bias Mitigation
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        mitigation.fit(labels, predictions, likelihoods, is_member)

        self.assertAlmostEqual(mitigation.p2p_prob_0, 0.8429378)
        self.assertAlmostEqual(mitigation.n2p_prob_0, 1.)
        self.assertAlmostEqual(mitigation.p2p_prob_1, 1.)
        self.assertAlmostEqual(mitigation.n2p_prob_1, 0.8893096)

    def test_numerical_stability_mixing_rate_large(self):

        # Randomly generated 50,000 data points with biased outcome
        df = self.generate_random_population(num_individuals=50000,
                                             threshold=0.5)

        # Data
        labels = df.label.to_numpy()
        predictions = df.y_pred_binary.to_numpy()
        likelihoods = df.prediction.to_numpy()
        is_member = df.group.to_numpy()

        # Bias Mitigation
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        mitigation.fit(labels, predictions, likelihoods, is_member)

        self.assertAlmostEqual(mitigation.p2p_prob_0, 0.819513)
        self.assertAlmostEqual(mitigation.n2p_prob_0, 1.)
        self.assertAlmostEqual(mitigation.p2p_prob_1, 0.644566)
        self.assertAlmostEqual(mitigation.n2p_prob_1, 1.)

    def test_numerical_stability_bias_mitigation(self):

        # Randomly generated 500 data points with biased outcome
        df = self.generate_random_population(num_individuals=500,
                                             threshold=0.5)

        np.random.seed(Constants.default_seed)

        # Randomly split the data into two sets, one for computing the mixing rate, and one evaluating the fairness
        order = np.random.permutation(len(df))
        train_indices = order[0::2]
        test_indices = order[1::2]
        train_data = df.iloc[train_indices].copy()
        test_data = df.iloc[test_indices].copy()

        # Mitigation object
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        # Train data
        train_labels = train_data.label.to_numpy()
        train_predictions = train_data.y_pred_binary.to_numpy()
        train_likelihoods = train_data.prediction.to_numpy()
        train_is_member = train_data.group.to_numpy()

        # Fit
        mitigation.fit(train_labels, train_predictions, train_likelihoods, train_is_member)

        # Test data
        test_labels = test_data.label.to_numpy()
        test_predictions = test_data.y_pred_binary.to_numpy()
        test_likelihoods = test_data.prediction.to_numpy()
        test_is_member = test_data.group.to_numpy()

        fair_predictions, fair_likelihoods = mitigation.transform(test_predictions, test_likelihoods, test_is_member)

        # Evaluate prior to mitigation
        test_data['y_pred'] = test_data['prediction'].apply(lambda x: (x > 0.5) * 1)

        fairness_metrics = BinaryFairnessMetrics().get_all_scores(test_data.label.tolist(),
                                                                  test_data.y_pred.tolist(),
                                                                  test_data.group.tolist()).reset_index()

        ao = fairness_metrics.loc[fairness_metrics.Metric == 'Average Odds', 'Value'].to_numpy().item()
        di = fairness_metrics.loc[fairness_metrics.Metric == 'Disparate Impact', 'Value'].to_numpy().item()
        eq = fairness_metrics.loc[fairness_metrics.Metric == 'Equal Opportunity', 'Value'].to_numpy().item()
        fnr = fairness_metrics.loc[fairness_metrics.Metric == 'FNR difference', 'Value'].to_numpy().item()
        gei = fairness_metrics.loc[fairness_metrics.Metric == 'Generalized Entropy Index', 'Value'].to_numpy().item()
        pe = fairness_metrics.loc[fairness_metrics.Metric == 'Predictive Equality', 'Value'].to_numpy().item()
        sp = fairness_metrics.loc[fairness_metrics.Metric == 'Statistical Parity', 'Value'].to_numpy().item()
        ti = fairness_metrics.loc[fairness_metrics.Metric == 'Theil Index', 'Value'].to_numpy().item()

        self.assertEqual(ao, -0.302)
        self.assertEqual(di, 0.579)
        self.assertEqual(eq, -0.285)
        self.assertEqual(fnr, 0.285)
        self.assertEqual(gei, 0.199)
        self.assertEqual(pe, -0.318)
        self.assertEqual(sp, -0.311)
        self.assertEqual(ti, 0.276)

        # Evaluate post mitigation
        fair_df = pd.DataFrame.from_dict({
            'label': test_labels,
            'group': test_is_member,
            'y_pred': (fair_likelihoods > 0.5) * 1})

        fairness_metrics = BinaryFairnessMetrics().get_all_scores(fair_df.label.tolist(),
                                                                  fair_df.y_pred.tolist(),
                                                                  fair_df.group.tolist()).reset_index()

        ao = fairness_metrics.loc[fairness_metrics.Metric == 'Average Odds', 'Value'].to_numpy().item()
        di = fairness_metrics.loc[fairness_metrics.Metric == 'Disparate Impact', 'Value'].to_numpy().item()
        eq = fairness_metrics.loc[fairness_metrics.Metric == 'Equal Opportunity', 'Value'].to_numpy().item()
        fnr = fairness_metrics.loc[fairness_metrics.Metric == 'FNR difference', 'Value'].to_numpy().item()
        gei = fairness_metrics.loc[fairness_metrics.Metric == 'Generalized Entropy Index', 'Value'].to_numpy().item()
        pe = fairness_metrics.loc[fairness_metrics.Metric == 'Predictive Equality', 'Value'].to_numpy().item()
        sp = fairness_metrics.loc[fairness_metrics.Metric == 'Statistical Parity', 'Value'].to_numpy().item()
        ti = fairness_metrics.loc[fairness_metrics.Metric == 'Theil Index', 'Value'].to_numpy().item()

        self.assertEqual(ao, 0.049)
        self.assertEqual(di, 1.072)
        self.assertEqual(eq, 0.060)
        self.assertEqual(fnr, -0.060)
        self.assertEqual(gei, 0.102)
        self.assertEqual(pe, 0.037)
        self.assertEqual(sp, 0.062)
        self.assertEqual(ti, 0.133)

        # Use fair predictions instead of fair likelihoods (get the same results)
        fair_df = pd.DataFrame.from_dict({
            'label': test_labels,
            'group': test_is_member,
            'y_pred': fair_predictions.astype(int)})

        fairness_metrics = BinaryFairnessMetrics().get_all_scores(fair_df.label.tolist(),
                                                                  fair_df.y_pred.tolist(),
                                                                  fair_df.group.tolist()).reset_index()

        ao = fairness_metrics.loc[fairness_metrics.Metric == 'Average Odds', 'Value'].to_numpy().item()
        di = fairness_metrics.loc[fairness_metrics.Metric == 'Disparate Impact', 'Value'].to_numpy().item()
        eq = fairness_metrics.loc[fairness_metrics.Metric == 'Equal Opportunity', 'Value'].to_numpy().item()
        fnr = fairness_metrics.loc[fairness_metrics.Metric == 'FNR difference', 'Value'].to_numpy().item()
        gei = fairness_metrics.loc[fairness_metrics.Metric == 'Generalized Entropy Index', 'Value'].to_numpy().item()
        pe = fairness_metrics.loc[fairness_metrics.Metric == 'Predictive Equality', 'Value'].to_numpy().item()
        sp = fairness_metrics.loc[fairness_metrics.Metric == 'Statistical Parity', 'Value'].to_numpy().item()
        ti = fairness_metrics.loc[fairness_metrics.Metric == 'Theil Index', 'Value'].to_numpy().item()

        self.assertEqual(ao, 0.049)
        self.assertEqual(di, 1.072)
        self.assertEqual(eq, 0.060)
        self.assertEqual(fnr, -0.060)
        self.assertEqual(gei, 0.102)
        self.assertEqual(pe, 0.037)
        self.assertEqual(sp, 0.062)
        self.assertEqual(ti, 0.133)

    def test_mitigation_fit_transform(self):

        # Randomly generated 500 data points with biased outcome
        df = self.generate_random_population(num_individuals=500,
                                             threshold=0.5)

        # Mitigation
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        # Train data
        labels = df.label.to_numpy()
        predictions = df.y_pred_binary.to_numpy()
        likelihoods = df.prediction.to_numpy()
        is_member = df.group.to_numpy()

        np.random.seed(Constants.default_seed)

        fair_predictions, fair_likelihoods = mitigation.fit_transform(labels,
                                                                      predictions,
                                                                      likelihoods,
                                                                      is_member)

        fair_df = pd.DataFrame.from_dict({
            'label': labels,
            'group': is_member,
            'y_pred': (fair_likelihoods > 0.5) * 1})

        fairness_metrics = BinaryFairnessMetrics().get_all_scores(fair_df.label.tolist(),
                                                                  fair_df.y_pred.tolist(),
                                                                  fair_df.group.tolist()).reset_index()

        ao = fairness_metrics.loc[fairness_metrics.Metric == 'Average Odds', 'Value'].to_numpy().item()
        di = fairness_metrics.loc[fairness_metrics.Metric == 'Disparate Impact', 'Value'].to_numpy().item()
        eq = fairness_metrics.loc[fairness_metrics.Metric == 'Equal Opportunity', 'Value'].to_numpy().item()
        fnr = fairness_metrics.loc[fairness_metrics.Metric == 'FNR difference', 'Value'].to_numpy().item()
        gei = fairness_metrics.loc[fairness_metrics.Metric == 'Generalized Entropy Index', 'Value'].to_numpy().item()
        pe = fairness_metrics.loc[fairness_metrics.Metric == 'Predictive Equality', 'Value'].to_numpy().item()
        sp = fairness_metrics.loc[fairness_metrics.Metric == 'Statistical Parity', 'Value'].to_numpy().item()
        ti = fairness_metrics.loc[fairness_metrics.Metric == 'Theil Index', 'Value'].to_numpy().item()

        self.assertEqual(ao, 0.056)
        self.assertEqual(di, 1.059)
        self.assertEqual(eq, 0.046)
        self.assertEqual(fnr, -0.046)
        self.assertEqual(gei, 0.091)
        self.assertEqual(pe, 0.066)
        self.assertEqual(sp, 0.052)
        self.assertEqual(ti, 0.113)

    def test_fit_transform_no_labels(self):

        # Data
        labels = None
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        # Bias Mitigation
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        with self.assertRaises(ValueError):
            mitigation.fit_transform(labels, predictions, likelihoods, is_member)

    def test_fitted_error_not_raised(self):

        # Data
        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        # Bias Mitigation
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        # Training: Learn mixing rates from labelled data
        mitigation.fit(labels, predictions, likelihoods, is_member)

        mitigation._check_fitted_mitigation()

        # Transform
        _ = mitigation.transform(predictions, likelihoods, is_member)

    def test_check_mitigation_input_valid(self):

        # Data
        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        mitigation = BinaryMitigation.EqualizedOdds()

        _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

    def test_check_mitigation_input_invalid_shapes(self):
        mitigation = BinaryMitigation.EqualizedOdds()

        # Labels wrong shape
        labels = np.array([1, 1, 0, 1, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaises(InputShapeError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

        # Predictions wrong shape
        labels = np.array([1, 1, 0, 1, 0, 1])
        predictions = np.array([0, 0, 0, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaises(InputShapeError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

        # Likelihoods wrong shape
        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        is_member = np.array([0, 0, 0, 1, 1, 1])
        with self.assertRaises(InputShapeError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

        # Member wrong shape
        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1])
        with self.assertRaises(InputShapeError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

    def test_check_mitigation_input_invalid_likelihood(self):
        mitigation = BinaryMitigation.EqualizedOdds()

        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 1.7])
        is_member = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaises(ValueError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

    def test_check_mitigation_input_invalid_is_member(self):
        mitigation = BinaryMitigation.EqualizedOdds()

        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 1.7])
        is_member = np.array([0, 0, 0, 1, 1, 2])

        with self.assertRaises(ValueError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

    def test_check_mitigation_input_invalid_type(self):
        mitigation = BinaryMitigation.EqualizedOdds()

        # Labels
        labels = str([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1, 2])

        with self.assertRaises(TypeError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

        # Predictions
        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = tuple([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = np.array([0, 0, 0, 1, 1, 2])

        with self.assertRaises(TypeError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

        # Likelihoods
        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = dict(data=[0.2, 0.3, 0.4, 0.5, 0.6, 1.7])
        is_member = np.array([0, 0, 0, 1, 1, 2])

        with self.assertRaises(TypeError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

        # Member
        labels = np.array([1, 1, 0, 1, 0, 0])
        predictions = np.array([0, 0, 0, 1, 1, 1])
        likelihoods = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        is_member = 1

        with self.assertRaises(TypeError):
            _ = mitigation._check_input_mitigation(labels, predictions, likelihoods, is_member)

    def test_fitted_error_raised(self):
        # Bias Mitigation
        mitigation = BinaryMitigation.EqualizedOdds(seed=Constants.default_seed)

        # omit training and raise fitted error
        with self.assertRaises(NotFittedError):
            mitigation._check_fitted_mitigation()

    @staticmethod
    def generate_random_population(num_individuals, threshold):

        np.random.seed(Constants.default_seed)

        size = num_individuals // 2
        # use larger probability of drawing positive outcome for one group
        y_true_0 = np.random.choice([0, 1], size=size, p=[0.3, 0.7])
        y_true_1 = np.random.choice([0, 1], size=size, p=[0.6, 0.4])

        y_pred_0 = []
        y_pred_1 = []
        for _ in range(size):
            # draw from a normal with different mean for each group
            pred_group_0 = np.clip(np.random.normal(loc=0.7, scale=0.3), 0, 1)
            pred_group_1 = np.clip(np.random.normal(loc=0.4, scale=0.3), 0, 1)
            y_pred_0.append(pred_group_0)
            y_pred_1.append(pred_group_1)

        is_member_0 = [0 for _ in range(size)]
        is_member_1 = [1 for _ in range(size)]

        df_0 = pd.DataFrame.from_dict({'prediction': y_pred_0, 'label': y_true_0, 'group': is_member_0})
        df_1 = pd.DataFrame.from_dict({'prediction': y_pred_1, 'label': y_true_1, 'group': is_member_1})

        df = pd.concat([df_0, df_1], axis=0).sample(frac=1)
        # create binary outcome based on user-defined threshold
        df['y_pred_binary'] = (df.prediction > threshold) * 1

        return df
