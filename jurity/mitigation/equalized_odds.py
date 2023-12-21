# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Union

import cvxpy as cvx
import numpy as np
import pandas as pd

from jurity.mitigation.base import _BaseMitigation
from jurity.utils import Constants
from jurity.utils import InputShapeError, NotFittedError
from jurity.utils import check_binary, check_false, check_true, check_or_convert_numpy_array


class EqualizedOdds(_BaseMitigation):
    def __init__(self, seed=Constants.default_seed):

        super().__init__("Equalized Odds",
                         "This is a class with bias mitigation post-processing technique "
                         "called Equalized Odds.")

        # Add Seed
        self.seed = seed

        # Mixing rates to mitigate bias
        self.p2p_prob_0 = 0
        self.n2p_prob_0 = 0
        self.p2p_prob_1 = 0
        self.n2p_prob_1 = 0

    def __repr__(self):
        return "\n".join(["Bias Mitigation Instance",
                          "Mixing rates:",
                          "P2P group 0:\t%.3f" % self.p2p_prob_0,
                          "N2P group 0:\t%.3f" % self.n2p_prob_0,
                          "P2P group 1:\t%.3f" % self.p2p_prob_1,
                          "N2P group 1:\t%.3f" % self.n2p_prob_1])

    def fit(self,
            labels: Union[List, np.ndarray, pd.Series],
            predictions: Union[List, np.ndarray, pd.Series],
            likelihoods: Union[List, np.ndarray, pd.Series],
            is_member: Union[List, np.ndarray, pd.Series]):
        """
        Idea: Imagine two groups have different ROC curves.
        Find the convex hull such that any FPR, TPR pair can be satisfied by either
        protected-group-conditional predictor. This might not be possible without randomization [4]_.

        The output of this optimization is a tuple of four probabilities of flipping the likelihood
        of a positive prediction to achieve equal FPR & TPR across two groups. We can then apply
        these learned mixing rates on new unseen data to achieve fairer distributions of outcomes.

        Parameters
        ----------
        labels: Union[List, np.ndarray, pd.Series]
            Binary ground truth labels for the provided dataset (0/1).
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        likelihoods: Union[List, np.ndarray, pd.Series]
            Scores between 0 and 1 from some black-box classifier.
        is_member: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).

        Returns
        ----------
        None.

        References
        ----------
            .. [4] Hardt, Moritz and Price, Eric and Price, Eric and Srebro, Nati,
            "Equality of Opportunity in Supervised Learning, Advances in Neural Information Processing Systems 29, 2016.
        """

        # Check input validity
        labels, predictions, likelihoods, is_member = self._check_input_mitigation(labels, predictions, likelihoods,
                                                                                   is_member)

        # Subset each group
        group_0 = is_member == 0
        group_1 = is_member == 1

        # Variables for each group
        variables_0 = self._get_variables(labels, likelihoods, predictions, group_0)
        variables_1 = self._get_variables(labels, likelihoods, predictions, group_1)

        # Objective function
        error = self._get_objective(variables_0, variables_1)

        # Constraints
        constraints = self._get_constraints(variables_0, variables_1)

        # Problem setup
        prob = cvx.Problem(cvx.Minimize(error), constraints)

        # Solve
        prob.solve()

        # Save fairness probabilities
        self.p2p_prob_0 = variables_0["p2p"].value
        if isinstance(self.p2p_prob_0,np.ndarray):
            self.p2p_prob_0=self.p2p_prob_0.item()
        self.n2p_prob_0 = variables_0["n2p"].value
        if isinstance(self.n2p_prob_0,np.ndarray):
            self.n2p_prob_0=self.n2p_prob_0.item()
        self.p2p_prob_1 = variables_1["p2p"].value
        if isinstance(self.p2p_prob_1,np.ndarray):
            self.p2p_prob_1=self.p2p_prob_1.item()
        self.n2p_prob_1 = variables_1["n2p"].value
        if isinstance(self.n2p_prob_1,np.ndarray):
            self.n2p_prob_1=self.n2p_prob_1.item()

    def fit_transform(self,
                      labels: Union[List, np.ndarray, pd.Series],
                      predictions: Union[List, np.ndarray, pd.Series],
                      likelihoods: Union[List, np.ndarray, pd.Series],
                      is_member: Union[List, np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fit and transform methods on the current dataset.

        Parameters
        ----------
        labels: Union[List, np.ndarray, pd.Series]
            Binary ground truth labels for the provided dataset (0/1).
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        likelihoods: Union[List, np.ndarray, pd.Series]
            Scores between 0 and 1 from some black-box classifier.
        is_member: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).

        Returns
        ----------
        fair_predictions: np.ndarray
            Fairer predictions with closely matching FPR & TPR across groups
        fair_likelihoods: np.ndarray
            Fairer likelihoods with closely matching FPR & TPR across groups
        """
        labels, predictions, likelihoods, is_member = self._check_input_mitigation(labels, predictions, likelihoods,
                                                                                   is_member)

        if labels is not None:
            self.fit(labels, predictions, likelihoods, is_member)
        else:
            raise ValueError("You need to provide valid labels to use this method.")

        return self.transform(predictions, likelihoods, is_member)

    def transform(self,
                  predictions: Union[List, np.ndarray, pd.Series],
                  likelihoods: Union[List, np.ndarray, pd.Series],
                  is_member: Union[List, np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fairness probabilistic mixing rates to a new dataset.

        The idea here is to probabilistically flip a subset of likelihoods and labels in each group
        based on learned mixing rates so that we achieve fairer distribution of outcomes.

        There is a trade-off between fairness and accuracy of a classifier. In general, repairing fairness metrics
        results in lower accuracy, but the relationship is non-linear and data dependent.

        Parameters
        ----------
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        likelihoods: Union[List, np.ndarray, pd.Series]
            Scores between 0 and 1 from some black-box classifier.
        is_member: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).

        Returns
        ----------
        fair_predictions: np.ndarray
            Fairer predictions with closely matching FPR & TPR across groups
        fair_likelihoods: np.ndarray
            Fairer likelihoods with closely matching FPR & TPR across groups
        """

        # Check input validity and transform to numpy arrays where neccessary
        _, predictions, likelihoods, is_member = self._check_input_mitigation(None, predictions, likelihoods,
                                                                              is_member)

        # Check object as fitted
        self._check_fitted_mitigation()

        group_0 = is_member == 0
        group_1 = is_member == 1
        likelihoods_0 = likelihoods[group_0]
        likelihoods_1 = likelihoods[group_1]
        predictions_0 = predictions[group_0]
        predictions_1 = predictions[group_1]

        # Mitigation for Group 0
        fair_predictions_0, fair_likelihoods_0 = self._adjust_fairness(predictions_0, likelihoods_0,
                                                                       self.p2p_prob_0, self.n2p_prob_0)

        # Mitigation for Group 1
        fair_predictions_1, fair_likelihoods_1 = self._adjust_fairness(predictions_1, likelihoods_1,
                                                                       self.p2p_prob_1, self.n2p_prob_1)

        # Keep predictions and likelihoods in the same order as the input
        fair_predictions = predictions.copy()
        fair_likelihoods = likelihoods.copy()

        # Use subsetting to have output in identical order as input
        fair_predictions[group_0] = fair_predictions_0
        fair_predictions[group_1] = fair_predictions_1
        fair_likelihoods[group_0] = fair_likelihoods_0
        fair_likelihoods[group_1] = fair_likelihoods_1

        # Return fair predictions and likelihoods
        return fair_predictions, fair_likelihoods

    def _get_variables(self, labels, likelihoods, predictions, group):
        """
        Prepare variables for optimization for a single group.
        """

        # Subset using group indices
        labels = labels[group]
        likelihoods = likelihoods[group]
        flip_likelihoods = 1 - likelihoods
        predictions = predictions[group]

        # Calculate base rate: the proportion of positive outcomes
        base_rate = np.mean(labels)

        # Set up the 2 variables corresponding to 2 probabilities of flipping sign on likelihoods
        # of a random subset of observations in a given group (positive to negative and vice versa).
        # Note that while we have 4 variables, we actually have 2 non-degenerate ones
        # since FPR + TNR = 1 and FNR + TPR = 1 for any classifier.

        # Two flipping probabilities, also called mixing rates
        p2p = cvx.Variable(1)
        n2p = cvx.Variable(1)
        n2n = cvx.Variable(1)  # trivially equals to 1 - n2p
        p2n = cvx.Variable(1) # trivially equals to 1 - p2p

        # Baseline label-wise FNR, FPR, TPR, TNR for the group
        tpr, fpr, tnr, fnr = self._get_label_wise_rates(labels, predictions)

        # Create placeholders for new FNR, FPR, TPR, TNR based on to-be-optimized mixing rates
        # so that we achieve equalized odds (we shift around original FNR, FPR, TPR, TNR)
        opt_fpr = fpr * p2p + tnr * n2p
        opt_fnr = fnr * n2n + tpr * p2n

        # Set up per-observation whether an observation was
        # true negative/positive or false negative/positive
        per_obs_tn = np.logical_and(predictions == 0, labels == 0)
        per_obs_fn = np.logical_and(predictions == 0, labels == 1)
        per_obs_tp = np.logical_and(predictions == 1, labels == 1)
        per_obs_fp = np.logical_and(predictions == 1, labels == 0)

        false_negative = (likelihoods * per_obs_fn).mean() * n2n + \
                         (flip_likelihoods * per_obs_fn).mean() * n2p

        true_positive = (likelihoods * per_obs_tp).mean() * p2p + \
                        (flip_likelihoods * per_obs_tp).mean() * p2n

        # Below is the generalized false negative rate
        # E(x,y) ~ G_0 [1 - h_t(x) | y = 1], in other words this is weighted expectation.
        # It is weighted by the probabilities of flipping the likelihoods and
        # normalized by dividing with the base rate
        pn_given_p = (false_negative + true_positive) / base_rate

        false_positive = (likelihoods * per_obs_fp).mean() * p2p + \
                         (flip_likelihoods * per_obs_fp).mean() * p2n

        true_negative = (likelihoods * per_obs_tn).mean() * n2n + \
                        (flip_likelihoods * per_obs_tn).mean() * n2p

        # This is the generalized false positive rate
        # E(x,y) ~ G_0 [ h_t(x) | y = 0]
        # It is weighted by the probabilities of flipping the likelihoods
        # and normalized by dividing with the base rate
        pp_given_n = (false_positive + true_negative) / (1 - base_rate)

        return {"p2p": p2p,
                "n2p": n2p,
                "n2n": n2n,
                "p2n": p2n,
                "opt_fpr": opt_fpr,
                "opt_fnr": opt_fnr,
                "pn_given_p": pn_given_p,
                "pp_given_n": pp_given_n}

    @staticmethod
    def _get_objective(variables_0, variables_1):
        """
        Formulate the error term to optimize, note we use FNR instead of TPR as the ROC curve is
        defined by FPR, TPR pairs of points and we use 1 - FNR in all calculations to match TPR.
        """

        opt_fpr_group_0 = variables_0["opt_fpr"]
        opt_fnr_group_0 = variables_0["opt_fnr"]
        opt_fpr_group_1 = variables_1["opt_fpr"]
        opt_fnr_group_1 = variables_1["opt_fnr"]

        return opt_fpr_group_0 + opt_fnr_group_0 + opt_fpr_group_1 + opt_fnr_group_1

    @staticmethod
    def _get_constraints(variables_0, variables_1):
        """
        Formulate optimization constraints to match ROC curves of two groups.
        """
        p2p_0 = variables_0["p2p"]
        p2n_0 = variables_0["p2n"]
        n2p_0 = variables_0["n2p"]
        n2n_0 = variables_0["n2n"]
        pn_given_p_0 = variables_0["pn_given_p"]
        pp_given_n_0 = variables_0["pp_given_n"]

        p2p_1 = variables_1["p2p"]
        p2n_1 = variables_1["p2n"]
        n2p_1 = variables_1["n2p"]
        n2n_1 = variables_1["n2n"]
        pn_given_p_1 = variables_1["pn_given_p"]
        pp_given_n_1 = variables_1["pp_given_n"]

        constraints = [  # the probability of flips and non flips needs to add up to 1
            p2p_0 == 1 - p2n_0,
            n2p_0 == 1 - n2n_0,
            p2p_1 == 1 - p2n_1,
            n2p_1 == 1 - n2n_1,

            # lower and upper bounds for variables
            p2p_0 <= 1,
            p2p_0 >= 0,
            n2p_0 <= 1,
            n2p_0 >= 0,
            p2p_1 <= 1,
            p2p_1 >= 0,
            n2p_1 <= 1,
            n2p_1 >= 0,

            # the two constraints below are to ensure equalized odds
            # (the first two are RHS, the later two are LHS of the two contraints)
            pn_given_p_0 == pn_given_p_1,
            pp_given_n_0 == pp_given_n_1]

        return constraints

    def _adjust_fairness(self,
                         predictions: Union[List, np.ndarray, pd.Series],
                         likelihoods: Union[List, np.ndarray, pd.Series],
                         p2p_prob: float,
                         n2p_prob: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using previously calculated probabilities of flipping indices,
        the goal is to match ROC curves to a maximally overlapping convex hull
        of the two ROC curves (conditional on from which group the ROC curve is calculated).
        """

        # Make sure predictions/likelihoods are np.ndarray so that subsetting works
        error_msg = "Input type not allowed, use numpy array, Pandas Series, or lists."
        predictions = check_or_convert_numpy_array(predictions, error_msg)
        likelihoods = check_or_convert_numpy_array(likelihoods, error_msg)

        fair_predictions = predictions.copy()
        fair_likelihoods = likelihoods.copy()

        # Subset positive/negative predictions
        pos_indices = np.where(predictions == 1)[0]
        neg_indices = np.where(predictions == 0)[0]

        # Shuffle these indices randomly
        np.random.seed(self.seed)
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        # Based on the flipping probability, subset indices to flip the likelihoods
        p2n_flip_indices = pos_indices[: int(len(pos_indices) * (1 - p2p_prob))]
        fair_likelihoods[p2n_flip_indices] = 1 - fair_likelihoods[p2n_flip_indices]
        fair_predictions[p2n_flip_indices] = 1 - fair_predictions[p2n_flip_indices]

        n2p_flip_indices = neg_indices[: int(len(neg_indices) * n2p_prob)]
        fair_likelihoods[n2p_flip_indices] = 1 - fair_likelihoods[n2p_flip_indices]
        fair_predictions[n2p_flip_indices] = 1 - fair_predictions[n2p_flip_indices]

        return fair_predictions, fair_likelihoods

    @staticmethod
    def _get_label_wise_rates(labels, predictions):
        """
        This method does not use TPR, FPR, etc. in a traditional sense.
        Instead, a label-wise agreement of label with prediction is calculated and then averaged
        Note this is NOT consistent with the sklearn implementation of calculating the confusion matrix.
        """
        true_positive_rate = np.mean(np.logical_and(predictions == 1, labels == 1))
        false_positive_rate = np.mean(np.logical_and(predictions == 1, labels == 0))
        true_negative_rate = np.mean(np.logical_and(predictions == 0, labels == 0))
        false_negative_rate = np.mean(np.logical_and(predictions == 0, labels == 1))

        return true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate

    def _check_fitted_mitigation(self):
        """
        Checks whether the bias mitigation instance was fitted or not.
        We assume the mixing rates are initialized all at 0 so if not fitted yet, all of them will be zero.
        """
        mixing_rates = [self.p2p_prob_0, self.n2p_prob_0, self.p2p_prob_1, self.n2p_prob_1]

        check_false(all(mixing_rates) == 0, NotFittedError("Mitigation instance needs to be fitted first."))

    @staticmethod
    def _check_input_mitigation(labels, predictions, likelihoods, is_member):
        """
        Check the following aspects:
        1) whether input is or can be converted to numpy arrays
        2) labels, predictions and is_member need to be binary
        3) likelihoods need to be between 0 and 1
        4) all arrays need to be of the same length
        """

        if labels is not None:
            # Check labels can be converted to numpy array
            msg = "Input type not allowed for {name}, allowed are numpy array, Pandas Series, or lists."
            labels = check_or_convert_numpy_array(labels, msg)
            # Check labels are binary
            check_binary(labels)

        # Check predictions
        msg = "Input type not allowed for predictions, allowed are numpy array, Pandas Series, or lists."
        predictions = check_or_convert_numpy_array(predictions, msg)

        # Check predictions type
        check_binary(predictions)

        # Check likelihoods
        msg = "Input type not allowed for likelihoods, allowed are numpy array, Pandas Series, or lists."
        likelihoods = check_or_convert_numpy_array(likelihoods, msg)

        # Check likelihoods between 0 and 1
        check_true(all(likelihoods >= 0) and all(likelihoods <= 1),
                   ValueError("Likelihood can be only between 0 and 1."))

        # Check is_member
        msg = "Input type not allowed for is_member, allowed are numpy array, Pandas Series, or lists."
        is_member = check_or_convert_numpy_array(is_member, msg)

        # Check predictions type
        check_binary(is_member)

        # Check shapes match
        if labels is not None:
            check_true(labels.shape[0] == predictions.shape[0], InputShapeError("", "Input shapes do not match."))

        check_true(predictions.shape[0] == likelihoods.shape[0] and
                   likelihoods.shape[0] == is_member.shape[0],
                   InputShapeError("", "Input shapes do not match."))

        return labels, predictions, likelihoods, is_member
