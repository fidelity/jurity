# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from jurity.utils import check_inputs_validity, unique_multiclass_multilabel, check_true


class _BaseBinaryFairness:
    """
    Base class to hold properties of Binary Fairness Metrics.

    This module is not intended to be used directly, instead it declares the basic skeleton of binary fairness class
    together with a set of parameters that are common to every fairness metrics.

    It declares properties that sub-classes will initiate according to each fairness metric.
    Note that the properties ought to not be changed by the end user and are private accordingly.

        - ``name`` property to display name of the class
        - ``description`` property to display description of the class
        - ``lower_bound`` property to display lower bound of a fairness metric according to 80% rule.
        - ``upper_bound`` property to display upper bound of a fairness metric according to 80% rule.
        - ``_ideal_value`` property to display ideal value of a fairness metric assuming zero bias.

    Attributes
    ----------
    _name: str
        The name of the fairness metric.
    _description: str
        The description of the fairness metric.
    _lower_bound: float
        Lower bound on fairness metric value to be considered "fair" according to 80% rule.
    _upper_bound: float
        Upper bound on fairness metric value to be considered "fair" according to 80% rule.
    _ideal_value: float
        Ideal value of the fairness metric assuming zero bias.

    - Note: 80% rule means that a machine learning classifier allows a maximum 20% difference in positive classification
    across groups (e.g., male/female) when controlling for other covariates.
    """

    def __init__(self, name, description, lower_bound, upper_bound, ideal_value):
        self._name = name
        self._description = description
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._ideal_value = ideal_value

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def ideal_value(self):
        return self._ideal_value


class _BaseMultiClassMetric(_BaseBinaryFairness):
    """
    Base class for calculating MultiClassFairnessMetrics.

    This module is not intended to be used directly, instead it declares the basic skeleton of fairness sub-classes.
    It further declares methods that sub-classes will initiate according to a fairness metric of interest.

    - ``binary_score`` method to declare what binary fairness metric to calculate after One-Vs-Rest conversion
    - ``get_score`` method to retrieve an array of fairness metrics, one for each class/arm/category

    Attributes
    ----------
    list_of_classes: List
        A list of model classes.
        Default value is None, in which case 0/1 labels are assumed.
    """

    def __init__(self, name: str, description: str,
                 lower_bound: float, upper_bound: float, ideal_value: float,
                 list_of_classes: Optional[List] = None):
        super().__init__(name,
                         description,
                         lower_bound=lower_bound,
                         upper_bound=upper_bound,
                         ideal_value=ideal_value)
        if list_of_classes is None:
            self.list_of_classes = [0, 1]
        else:
            self.list_of_classes = list_of_classes

    @abc.abstractmethod
    def _binary_score(self, predictions: Union[np.ndarray, pd.Series, List],
                      is_member: Union[List, np.ndarray, pd.Series]) -> float:
        """
        Abstract method to call a binary fairness metric of choice.

        Parameters
        ----------
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        is_member: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).

        Returns
        ----------
        Fairness metric.
        """
        raise NotImplementedError()

    def get_scores(self, predictions: Union[List[List], np.ndarray, pd.Series, List],
                   is_member: Union[List, np.ndarray, pd.Series]) -> List[float]:
        """
        Method to calculate a fairness score for each class.

        Parameters
        ----------
        predictions: Union[List, np.ndarray, pd.Series]
            Binary predictions from some black-box classifier (0/1).
        is_member: Union[List, np.ndarray, pd.Series]
            Binary membership labels (0/1).

        Returns
        ----------
        Fairness metric for each class.
        """
        check_inputs_validity(predictions, is_member, optional_labels=True, binary_only=False)

        one_hot_predictions = self._one_hot_encode_classes(predictions)

        scores = []
        for class_ in self.list_of_classes:
            score = self._binary_score(one_hot_predictions[class_], is_member)
            scores.append(score)
        return scores

    def _one_hot_encode_classes(self, predictions: Union[List[List], np.ndarray, pd.Series, List]) -> pd.DataFrame:
        """
        Method to convert a series with a list of predicted classes for each row
        to a one-hot encoded representation to enable per-class fairness evaluation.
        """
        mlb = MultiLabelBinarizer(classes=self.list_of_classes)

        # Multi-label binarizer unfortunately does not work with a list of ints/floats
        # Hence, we need to convert class labels list[int/float] -> list[list[int/float]]
        if isinstance(predictions[0], int) or isinstance(predictions[0], float):
            predictions = [[i] for i in predictions]

        # Check that predictions only contain labels from self.list_of_classes
        unique_predictions = unique_multiclass_multilabel(predictions)

        check_true(len(set(unique_predictions).difference(set(self.list_of_classes))) == 0,
                   ValueError("Supplied predictions do not match unique class labels."))

        # Convert input to pandas series
        predictions = pd.Series(predictions)

        return pd.DataFrame(
            mlb.fit_transform(predictions),
            columns=mlb.classes_,
            index=predictions.index,
        )
