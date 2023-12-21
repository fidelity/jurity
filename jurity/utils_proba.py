# import warnings filter and ignore future warning due to pandas.loc
import warnings
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
from jurity.constants import Constants
from jurity.utils import InputShapeError, WeightTooLarge
from jurity.utils import check_true, check_binary, check_input_type, check_input_1d, check_elementwise_input_type
from typing import List, Union


def get_argmax_memberships(memberships, membership_labels):
    is_member = []
    for likelihoods in memberships:
        max_index = likelihoods.index(max(likelihoods))
        is_protected = 1 if max_index in membership_labels else 0
        is_member.append(is_protected)
    return is_member


def check_memberships_proba(memberships, len_predictions, unique_surrogate_list):
    """
    Make sure probabilistic memberships are a 2D list or array with the right dimensions
    """
    check_input_type(memberships)
    len_likelihoods = len(memberships[0])
    check_true(len(memberships) == len_predictions,
               InputShapeError("", f"Likelihoods outer array/list must be same length as predictions array. "
                                   f"Likelihood array is:{len_likelihoods}. Predictions array is: {len_predictions}"))

    for i, likelihood in enumerate(memberships):
        check_true(type(likelihood) in [np.ndarray, list],
                   TypeError(f"Membership likelihoods need to be 2D lists or arrays. "
                             f"Likelihood at {i} is not array or list."))

        # Size match, for inner array (all arrays should be same size)
        len_this_like = len(likelihood)
        check_true(len_likelihoods == len_this_like,
                   InputShapeError("",
                                   f"Shapes of inputs do not match. "
                                   f"Number of classes: {len_this_like}"
                                   f"You supplied array lengths "
                                   f"size: {len_likelihoods}, at index: {i}."))
        check_true(math.isclose(np.sum(likelihood), 1.0),
                   InputShapeError("", "Membership likelihood does not sum to 1.0. "
                                       "Sums to {0} at index {0}.".format(np.sum(likelihood), i)))

    # Likelihoods must either match the length of the predictions vector
    # or be a pandas dataframe with a unique index for the surrogate classes


def check_memberships_proba_df(memberships_df: pd.DataFrame, unique_surrogate_list: set, membership_names: List[str]):
    if membership_names is None:
        membership_names = memberships_df.columns
    sum_to_one = pd.Series(memberships_df.sum(axis=1)).apply(lambda x: math.isclose(x, 1.0))
    check_true(len(unique_surrogate_list) <= memberships_df.shape[0],
               InputShapeError("", "Memberships dataframe must have one row per surrogate class."))
    check_true(unique_surrogate_list.issubset(memberships_df.index.values),
               InputShapeError("", "Memberships dataframe must have an index with surrogate values"))
    check_true(memberships_df.shape[1] == len(membership_names),
               InputShapeError("", "Memberships dataframe must have one column per protected class name."))
    # Make sure each row in the input dataframe sums to 1.
    check_true(np.all(sum_to_one), ValueError("Each row in membership dataframe must sum to 1."))


def check_inputs_proba(predictions: Union[List, np.ndarray, pd.Series],
                       memberships: Union[List[List], np.ndarray, pd.Series, pd.DataFrame],
                       surrogates: Union[List, np.ndarray, pd.Series],
                       membership_labels: Union[int, float, str, List[int]],
                       membership_names: List[str] = None,
                       must_have_labels: bool = False,
                       labels: Union[List, np.ndarray, pd.Series] = None):
    check_input_type(surrogates)

    len_surrogate_class = len(surrogates)
    len_predictions = len(predictions)
    check_true(len_predictions == len(surrogates),
               InputShapeError("", f"Shapes of inputs do not match. "
                                   f"You supplied array lengths "
                                   f" predictions: {len_predictions}."
                                   f"surrogate_class: {len_surrogate_class}"))

    # Need protected class likelihoods for non-binary/surrogate membership
    check_true(memberships is not None,
               ValueError("For non-binary membership, need to provide membership likelihoods"))

    if isinstance(memberships, pd.DataFrame):
        check_memberships_proba_df(memberships, set(surrogates), membership_names)
        len_likelihoods = memberships.shape[1]
    else:
        check_memberships_proba(memberships, len_predictions, set(surrogates))
        len_likelihoods = len(memberships[0])

    # Protected label is bounded by the number of protected
    if isinstance(membership_labels, list):
        check_true(len(membership_labels) < len_likelihoods,
                   ValueError("Protected label must be less than number of classes"))


def get_bootstrap_results(predictions: Union[List, np.ndarray, pd.Series],
                          memberships: Union[List, np.ndarray, pd.Series, List[List], pd.DataFrame],
                          surrogates: Union[List, np.ndarray, pd.Series],
                          membership_labels: Union[str, int, List[str], List[int]],
                          labels: Union[List, np.ndarray, pd.Series] = None,
                          bootstrap_trials: int = Constants.bootstrap_trials,
                          membership_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate bootstrap results for surrogate class analysis.
    This function detects input types, checks inputs for correctness, and returns a Pandas dataframe with building blocks.
    It is intended to be flexible with appropriate warnings when inputs will lead to erroneous conclusions.

    Parameters:
        predictions: : Predictions from a binary classifier.
            1-dimensional array, list or pandas.Series with length == n individuals predicted.
        memberships: Class likelihoods based on surrogate membership.
            pandas.DataFrame with shape=(n surrogate classes, n classes) and index=surrogate class
            --or--
            2-dimensional array with shape=(number of individuals predicted,n classes), or
            pandas.Series of lists with length=n individuals predicted and each inner list having length=n classes
            Each inner list must sum to 1.
        surrogates: Surrogate class membership for each individual.
            1-dimensional array, list, or pandasSeries with length=n individuals predicted.
        membership_labels: Union[str, int, List[str], List[int]],
            a list, array, or pandas Series listing which of the classes are considered protected
            length must be less than the number of classes
        labels: Union[List, np.ndarray, pd.Series] = None,
            optional list, array, or pandas Series with ground truth labels for each individual.
            length=number of individuals predicted.
        bootstrap_trials: int = Constants.bootstrap_trials,
            number of bootstraps to be run. Default is 100, which should be enough for most cases and run quickly
        membership_names: List[str] = None) -> pd.DataFrame:
            optional list of labels for class membership. Used to label output Dataframe
"""
    # Z is the number of unique values for the surrogate variable.
    # p is the length of the interior vectors of memberships

    # Summarize input data by surrogate variable to create the following:
    # Y [Z by 4]: A summary of the confusion matrix, with percentage of individuals in each quadrant
    #   Using predictions and labels to calculate the confusions matrix quadrant for each individual
    #   summarize a surrogate level as true_positive_ratio, false_positive_ration, true_negative_ratio, false_negative_ratio
    #   this lets us calculate inferred values for protected and unprotected groups
    #   Note: If labels are not available, Y will be [Z by 1] with only the percent_positive_ratio
    # X [Z by p]: Likelihoods at surrogate variable level, p>=2 with the probabilities of protected and unprotected status
    # W [Z by 1]: The number of individuals for each level of Z
    # Right now, these are returned as a single dataframe
    if membership_names is None:
        if isinstance(memberships, pd.DataFrame):
            membership_names = memberships.columns.to_list()
        else:
            membership_names = ["A", "B"]

    if labels is None:
        check_inputs_proba(predictions, memberships, surrogates, membership_labels,
                           membership_names=membership_names)
    else:
        check_inputs_proba(predictions, memberships, surrogates, membership_labels,
                           membership_names=membership_names, must_have_labels=True, labels=labels)

    summary_df = SummaryData.summarize(predictions, memberships, surrogates, labels, membership_names)

    # Add X, Y, and W matricies to the BiasCalculator
    if labels is not None:
        bc = BiasCalculator.from_df(summary_df, membership_labels, membership_names)
    else:
        bc = BiasCalculator.from_df(summary_df, membership_labels, membership_names,
                                    test_names=[Constants.prediction_ratio])

    # Run bootstrapping to calculate inferred metrics
    # for 1 to 100:
    #   sample Z rows from X, Y, and W with replacement
    #   Calculate the WOLS estimates for:
    #       true_positive_ratio, false_positive_ration, true_negative_ratio, false_negative_ratio for protected and unprotected groups
    #       Calculate TPR, FNR, FPR, and TNR for the sample.
    #       Store the estimates.
    bootstrap_df = bc.run_bootstrap(bootstrap_trials)

    # Calculate the means of all statistics across the 100 bootstrapped samples
    # Calculate average odds, equal opportunity, predictive equality, false negative difference, statistical parity
    transformed_bootstrap = bc.transform_bootstrap_results(bootstrap_df)
    return transformed_bootstrap


def unpack_bootstrap(df: pd.DataFrame, stat_name: str, membership_labels: List[int]):
    """
    For Binary classifiers, return the requested model performance statistics for each class.
    Currently only implemented for cases where there are two classes: Protected and unprotected.
    Arguments:
        df: input df
        stat_name: Name of the requested statistic, e.g. FNR for false negative rate.
        membership_labels: list of classes requesting membership for.
    """
    stats = df[[stat_name]]
    v = stats.index.values
    if len(v) != 2 or len(membership_labels) > 1:
        raise ValueError("Unpacking for probabilistic results only enabled for binary metrics.")
    return stats.loc[v[membership_labels], stat_name][0], stats.loc[np.delete(v, membership_labels), stat_name][0]


class BiasCalculator:
    """
    Encapsulates raw inputs for bootstrap calculations and classes to calculate and store bootstrap results
    Members:
    _y: measurements to be calculated
    _x: Input race percentages
    _w: Weights for weighted regression
    _prediction_matrix: Matrix needed to predict model diagnostic values for different classes
    _class_labels: Labels for class membership likelihood columns
    _test_labels: Labels for model performance diagnostics
    """

    @classmethod
    def from_df(cls,
                df: pd.DataFrame,
                membership_labels: List[int],
                membership_names: List[str],
                test_names: List[str] = None,
                weight_name: str = Constants.weight_col_name,
                weight_warnings: bool = True):
        """
        Reads an input dataframe and returns a BiasCalculator
        Arguments:
            df: a pandas DataFrame with shape=(surrogate_class,n classes).
                Columns are class probabilities and summaries of test statistics for each surrogate class.
            membership_labels: List[int], which membership_names are in the protected group
            membership_names: List[str], names of classes, should correspond to columns in input dataframe
            test_names:
            weight_name: Name of column containing weight variable Default is Constants.weight_col_name.
            weight_warnings: If false suppresses warnings about small counts for surrogate classes
        """
        if test_names is None:
            test_names = [Constants.true_positive_ratio, Constants.true_negative_ratio,
                          Constants.false_positive_ratio, Constants.false_negative_ratio,
                          Constants.prediction_ratio]

        if (np.any([m < 0 for m in membership_labels]) or
                np.any([m >= len(membership_names) for m in membership_labels])):
            raise ValueError(f"Protected membership_label:{membership_labels} not in "
                             f"membership_names:{membership_names}.")

        bcdf = BiasCalcFromDataFrame(membership_names, weight_name, membership_labels, test_names)

        # Start with min_weight set at the recommended default.
        # If it's too high, set at 10.
        # If 10 is too high, set at 1 and give a warning.
        try:
            bc = bcdf.get_bias_calculator(df, weight_warnings=weight_warnings)
        except WeightTooLarge:
            try:
                bc = bcdf.get_bias_calculator(df, 10, weight_warnings=weight_warnings)
            except WeightTooLarge:
                bc = bcdf.get_bias_calculator(df, 1, weight_warnings=weight_warnings)
        return bc

    def __init__(self, Y, X, W, class_labels, test_labels, verbose=True):
        self.Y(Y)
        self.X(X)
        self.W(W)
        if class_labels:
            self.class_labels(class_labels)
        if test_labels:
            self.test_labels(test_labels)
        self.check_dimensions()
        self.verbose(verbose)

    def verbose(self, value=None):
        if value is not None:
            self._verbose = value
        return self._verbose

    def Y(self, value=None):
        """
        Get or set input value as a numpy array that has 2 dimensions.
        Arguments:
            value: 2-D numpy array to be set
        """
        if value is not None:
            try:
                y = np.array(value, dtype='f')
            except ValueError:
                print("Cannot convert Y input to a numpy array of floats.")
                return None
            if y.ndim == 1:
                self._y = np.reshape(y, (-1, y.shape[0]))
            elif not y.ndim == 2:
                raise ValueError("Input Y must have 1 or 2 dimensions.")
            self._y = y
            # always update n_tests when we update Y
            self._n_tests = self._y.shape[1]
        return self._y

    def X(self, value=None):
        """
        Get or set input x as a 2-D numpy array.
        Arguments:
            value: value to be set, matrix of membership probabilities, shape-(n surrogate classes, n classes-1)
        """
        if value is not None:
            try:
                x = np.array(value, dtype='f')
            except ValueError:
                print("Cannot convert X into a numpy array of floats.")
            if x.ndim == 1:
                self._x = np.reshape(x, (-1, x.shape[0]))
            elif not x.ndim == 2:
                raise ValueError("Input X must have 1 or 2 dimensions")
            self._x = x
            self.prediction_matrix(construct=True)
        return self._x

    def W(self, value=None):
        """
        Get or set input weight vector w as a numpy array that has 1 dimension
        Arguments:
            value: np.ndarray with length==n surrogate classes, containing the count of individuals in each class
        """
        if value is not None:
            try:
                w = np.array(value, dtype='f')
            except ValueError:
                print("Cannot convert W into a numpy array or floats.")
            if not w.ndim == 1:
                raise ValueError("W must be a one-dimensional array.")
            self._w = w
        return self._w

    def class_labels(self, value=None):
        """
        Set and get labels for class membership probabilities.
        Arguments:
            value: a 2D list of lists containing strings, form[["str1"],["str2","str3",...]]
        """
        r_form = "Surrogate labels list must be of form [[omitted category],[race_1, race_2,...]]"
        input_form = f"Input was of form {value}"
        if value:
            if not isinstance(value, list):
                raise ValueError(f"{r_form}\n{input_form}")
            elif not len(value) == 2:
                raise ValueError(f"{r_form}\n{input_form}")
            elif not len(value[0]) == 1:
                raise ValueError("There can only be one omitted category")
            else:
                if len(value[1]) != self._x.shape[1]:
                    raise ValueError(
                        "Surrogate labels does not match dimension of X. Update X first, or check dimensions of input labels.")
                else:
                    self._compare_label = value[0][0]
                    self._non_compare_labels = value[1]
                    self.all_class_labels(calculate=True)
        return [[self._compare_label], self._non_compare_labels]

    def all_class_labels(self, calculate=False):
        """
        Get or set all surrogate labels based on _compare_label and _non_compare_labels.
        Used to index bootstrap output.
        Arguments:
            calculate: whether labels should be re-created. If False uses already stored values.
        """
        if calculate:
            l = [self._compare_label]
            for c in self._non_compare_labels:
                l.append(c)
            self._all_class_labels = l
        return self._all_class_labels

    def test_labels(self, value: List[str] = None):
        """
        Get or set names of test columns.
        Arguments:
            value: list of strings equal to number of columns in _y
        """
        if value is not None:
            if not isinstance(value, list):
                raise ValueError(f"test_labels must be a list of strings. Input: {value}")
            else:
                for s in value:
                    if not isinstance(s, str):
                        raise ValueError(f"test_labels must be a list of strings. Input: {value}")
            self._test_labels = value
        return self._test_labels

    def prediction_matrix(self, construct=False):
        """
        Build and Return the matrix used to calculate the predicted statistics for each class
        Arguments:
            construct: If True, re-creates prediction matrix. Called when _x is updated.
        """
        # Construct matrix of all possible values, corresponding to class_labels()
        if construct:
            n_xs = self.X().shape[1]
            self._prediction_matrix = np.concatenate((np.zeros((1, n_xs)), np.identity(n_xs)))
        return self._prediction_matrix

    def check_dimensions(self):
        """
        When change are made, check the dimensions of X, Y, and W to make sure they still match.
        """
        if not self.Y().shape[0] == self.X().shape[0]:
            raise ValueError(
                "Dimensions of X and Y do not match. Dimensions of X:- {0}, Dimensions of Y: {1}".format(self._x.shape,
                                                                                                         self._y.shape))
        elif not self.Y().shape[0] == self.W().shape[0]:
            raise ValueError("Length of W does not match X and Y")
        elif not self.Y().shape[1] == len(self.test_labels()):
            raise ValueError("Y and test_names have different dimensions. Y:{0} labels: {1}.".format(self.Y().shape[1],
                                                                                                     len(self.test_labels())))
        elif not self._prediction_matrix.shape == (self.X().shape[1] + 1, self.X().shape[1]):
            # If we are checking the dimensions, the problem could be that tht prediction matrix hasn't been constructed yet
            self.prediction_matrix(construct=True)
            if not self._prediction_matrix.shape == (self.X().shape[1] + 1, self.X().shape[1]):
                raise ValueError(
                    "Prediction matrix should be square with dim [1 + number of cols in X,n cols in Y]. X dim: {0}. Prediction matrix has dimensions: {1}" \
                        .format(self.X().shape, self.prediction_matrix().shape))
        else:
            return True

    def calc_one_bag(self, in_X: np.ndarray,
                     in_Y: np.ndarray,
                     in_W: np.ndarray) -> dict:
        """
        Calculate the regression of all surrogate-based tests
        Arguments:
            in_X: The sampled X for this bootstrap
            in_Y: The sampled Y for this bootstrap
            in_W: The sampled weights for this bootstrap
        return: a dictionary {str: []}
                where keys are equal to the tests calculated and values equal to the predicted value for each class
        """
        all_models = {}
        names_of_Ys = self.test_labels()
        # Run one regression
        for i in range(len(names_of_Ys)):
            y = in_Y[:, i]
            m = LinearRegression()
            m.fit(in_X, y, in_W)
            m.predict(self._prediction_matrix)
            all_models[names_of_Ys[i]] = m.predict(self._prediction_matrix)
        return all_models

    def run_bootstrap(self, bootstrap_trials: int) -> pd.DataFrame:
        """
        Collect data for final calculation using the bootstrap
        Returns a dataframe with a column for each calculation and a row for each bootstrap sample
        Inputs: n_boots: Number of bootstrapped samples
        """
        all_model_results = []
        n_rows = self.X().shape[0]
        for i in range(bootstrap_trials):
            select_these = np.random.choice(range(0, n_rows), n_rows)
            in_X = self.X()[select_these]
            in_Y = self.Y()[select_these]
            in_W = self.W()[select_these]

            # calc_one_bag modified so we return the predictions instead of the coefficients.
            preds = pd.DataFrame.from_dict(self.calc_one_bag(in_X, in_Y, in_W))

            # FPR, FNR, etc are functions of stats we calculated.
            # We have to calculate individually and then take the grand mean.
            binary_metrics = self.add_binary_metrics(preds)
            if binary_metrics is not None:
                all_model_results.append(pd.concat([binary_metrics, preds], axis=1))
            else:
                preds['class'] = self.all_class_labels()
                all_model_results.append(preds)
        out_data = pd.concat(all_model_results, axis=0).reset_index().drop(["index"], axis=1)
        return out_data

    def add_binary_metrics(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        """
        Add False Positive Rate, False Negative Rate, True Positive Rate and True Negative Rate to a set of bootstrap predictions.
        These have to be added at the bootstrap level, before means are calculated, so they have the right expected values
        """
        tests_we_have = df.columns
        # For binary classifiers, if we know the true labels, we will probably want these tests.
        common_tests = [Constants.false_positive_ratio, Constants.false_negative_ratio, Constants.true_positive_ratio,
                        Constants.true_negative_ratio]
        out_cols = []
        if set(common_tests).issubset(set(tests_we_have)):
            FPR = df[Constants.false_positive_ratio] / (
                    df[Constants.false_positive_ratio] + df[Constants.true_negative_ratio])
            FPR.name = Constants.FPR
            out_cols.append(FPR)

            FNR = df[Constants.false_negative_ratio] / (
                    df[Constants.false_negative_ratio] + df[Constants.true_positive_ratio])
            FNR.name = Constants.FNR
            out_cols.append(FNR)

            TPR = df[Constants.true_positive_ratio] / (
                    df[Constants.true_positive_ratio] + df[Constants.false_negative_ratio])
            TPR.name = Constants.TPR
            out_cols.append(TPR)

            TNR = df[Constants.true_negative_ratio] / (
                    df[Constants.true_negative_ratio] + df[Constants.false_positive_ratio])
            TNR.name = Constants.TNR
            out_cols.append(TNR)

            ACC = df[Constants.true_positive_ratio] + df[Constants.true_negative_ratio]
            ACC.name = Constants.ACC
            out_cols.append(ACC)
        # For binary classifiers we can always calculate the prediction_ratio, even if we don't have anything else
        if Constants.prediction_ratio in tests_we_have:
            prediction_rate = df[Constants.prediction_ratio]
            prediction_rate.name = Constants.PRED_RATE
            out_cols.append(prediction_rate)
        if len(out_cols) > 0:
            df = pd.concat(out_cols, axis=1)
            df['class'] = self.all_class_labels()
            return df
        else:
            return None

    @staticmethod
    def transform_bootstrap_results(df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes means of inputs from raw bootstrap results.
        Input: Pandas Dataframe that is the result of self.run_bootstrap
        Returns: means of all statistics grouped by class
        """
        return df.groupby(Constants.class_col_name).mean()

    def __str__(self):
        return "BiasCalculator(class_labels=" + str(self.class_labels()) + ", test_labels=" + str(
            self.test_labels()) + ")"


class BiasCalcFromDataFrame:
    """
    Class that creates a bias calculator from a dataframe.
    Members:
    _class_names: Names of surrogate class labels
    _weight_name: Name of column with weights
    _compare_label: label of group that's comparison group from the regression
    _test_names: Names of tests to be calculated
    """

    def __init__(self, membership_names: List[str],
                 weight_name: str,
                 membership_labels: List[int],
                 test_names: List[str]):
        """
        Initialize names to be read and name of comparison category for regression.
        Members:
        _class_names: Names of classes, should be columns in pandas dataframe
        _compare_label: Name of comparison group, usually the unprotected class
            if multiple unprotected classes, we use the first one.
        _
        """

        omitted_string = None
        # Get the first non-protected group listed.
        for idx, name in enumerate(membership_names):
            if idx not in membership_labels:
                omitted_string = name
                break
        if omitted_string is None:
            raise ValueError(
                "All groups appear to be protected groups. Must designate at least one non-protected group.")
        self.compare_label(omitted_string)
        self.class_names(membership_names)
        if self._compare_label in membership_names:
            self._class_names.remove(self._compare_label)
        self.test_names(test_names)
        self.weight_name(weight_name)

    def class_names(self, value: List[str] = None):
        """
        Get or set surrogate class names. Make sure it is a list of strings
        """
        if value is not None:
            if type(value) != list:
                raise ValueError("Surrogate class names must be a list of strings")
            v = list(set(value))
            for l in v:
                if not isinstance(l, str):
                    raise ValueError(f"Surrogate class name {l} is not a string.")
            self._class_names = value
            if not len(self._class_names) == len(v):
                raise ValueError("Surrogate class name contains duplicates.")
        return self._class_names

    def test_names(self, value: List[str] = None):
        """
        Get or set names of test columns. Make sure it is a list of strings
        """
        if value:
            if type(value) != list:
                raise ValueError("Test names must be a list of strings")
            v = list(set(value))
            for l in v:
                if not isinstance(l, str):
                    raise ValueError(f"Test name {l} is not a string.")
            self._test_names = value
            if not len(self._test_names) == len(v):
                raise (ValueError, "List of test names contains duplicates.")
        return self._test_names

    def compare_label(self, value: str = None):
        """
        Get or set compare group. Make sure it is a single string.
        """
        if value:
            e = "Name of omitted category is {0}, not a string."
            if self.check_single_string(value, e):
                self._compare_label = value
        return self._compare_label

    def pred_name(self, value: str = None):
        """
        Get or set name of column with predictions. Make sure it is a single string.
        """
        if value:
            e = "Name of predicted column is {0}, not a string."
            if self.check_single_string(value, e):
                self._pred_name = value
        return self._pred_name

    def true_name(self, value: str = None):
        """
        Get or set name of column with true classifications. Make sure it is a single string.
        """
        if value:
            e = "Name of predicted column is {0}, not a string."
            if self.check_single_string(value, e):
                self._true_name = value
        return self._true_name

    def weight_name(self, value: str = None):
        """
        Get or set name of weight column. Make sure it is a single string.
        """
        if value:
            e = "Name of weight column is {0}, not a string."
            if self.check_single_string(value, e):
                self._weight_name = value
        return self._weight_name

    @staticmethod
    def check_single_string(value, error_msg):
        """
        Helper function to check inputs that are supposed to be single strings
        """
        if not isinstance(value, str):
            raise ValueError(error_msg.format(value))
        else:
            return True

    def get_X_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make X matrix for bias calculator.
        Arguments
            df: pd.DataFrame, summarized by surrogate class, that has class membership probabilities for each surrogate class.
        """
        if not set(self.class_names()).issubset(set(df.columns)):
            raise ValueError(
                "Surrogate names: {0} are not in dataframe.".format(set(self._class_names) - (set(df.columns))))
        return df[self.class_names()].to_numpy(dtype='f')

    def get_Y_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make Y matrix for bias calculator.
        Arguments:
            df: pd.DataFrame, summarized by surrogate class, that has the summarized model performance statistics
        """
        if not set(self._test_names).issubset(set(df.columns)):
            raise ValueError("Test names: {0} are not in dataframe.".format(set(self._test_names) - (set(df.columns))))
        return df[self._test_names].to_numpy(dtype='f')

    def get_W_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make W array for bias calculator.
        Arguments:
            df: pd.DataFrame, summarized by surrogate class value, that has the weight column.
        """
        if not self._weight_name in set(df.columns):
            raise ValueError("weight name: {0} are not in dataframe.".format(self._weight_name))
        return df[self._weight_name].to_numpy(dtype='f')

    def get_bias_calculator(self, df: pd.DataFrame, min_weight: int = 0, weight_warnings: bool = True):
        """
        Make bias calculator.
        Arguments:
            df: pd.DataFrame, summarized by surrogate class, with columns for confusion matrix and/or prediction percentages
            min_weight: surrogate classes that are smaller than this value will be dropped.
            weight_warnings: Whether to print warnings when too many rows are dropped from surrogate class matrix
        """
        if min_weight < 5:
            if weight_warnings:
                warnings.warn("Recommended minimum count for surrogate class is 5. "
                              "Minimum weights of less than 5 will give unstable results.")

        if self.weight_name() in df.columns:
            subset = df[df[self._weight_name] >= min_weight]
            if weight_warnings:
                print("{0} rows removed from datafame for insufficient weight values" \
                      .format(df.shape[0] - subset.shape[0]))
            if subset.shape[0] < len(self.class_names()):
                raise WeightTooLarge("Input dataframe does not have enough rows to estimate surrogate classes "
                                     "reduce minimum weight.")
        else:
            raise ValueError("Weight variable {0} is not in dataframe.".format(self.weight_name()))

        # Create Bias Calculator
        X = self.get_X_matrix(subset)
        Y = self.get_Y_matrix(subset)
        W = self.get_W_array(subset)
        bc = BiasCalculator(Y, X, W, [[self.compare_label()], self.class_names()], self.test_names())

        return bc

    def __str__(self):
        return "BiasCalculatorFromDataFrame(class_names=" + str(self.class_names()) + ", test_names=" + str(
            self.test_names()) + ")"


class SummaryData:
    """
    Class that calculates summary data by surrogate class from input detailed data that has surrogate class columns.
    Members:
        _tests: Names of tests to be calculated
        _surrogate_perf_col_name: Name of surrogate column in performance data
        _surrogate_surrogate_col_name: Name of surrogate column in surrogate class data
        _pred_name: Name of column with predicted label
        _true_name: Name of column with true label (optional)
    """

    @classmethod
    def summarize(cls,
                  predictions: Union[List, np.ndarray, pd.Series],
                  memberships: Union[List, np.ndarray, pd.Series, pd.DataFrame],
                  surrogates: Union[List, np.ndarray, pd.Series],
                  labels: Union[List, np.ndarray, pd.Series] = None,
                  membership_names: List[str] = None,
                  warnings:bool = False) -> pd.DataFrame:
        """
        Return a summary dataframe suitable for bootstrap calculations.
        Arguments:
            predictions: List, array, or pd.Series with predicted outcome for each individual.
            memberships: pd.DataFrame, indexed by surrogate class, with membership probabilities for each surrogate class
            -or-
            List, array, or pd.Series with inner list, array, or eries of floats containing class probabilities for each individual
            surrogates: Surrogate class for each individual
            labels: Optional list, array, or pd.Series of actual outcome for each individual. If None, a subset of fairness metrics are returned
            membership_names: Optional list of strings, where len() is equal to length of inner array of membership probabilities. Used to provide labels for outcomes.
        """
        if membership_names is None:
            membership_names = ["A", "B"]

        df = pd.concat([pd.Series(predictions, name="predictions"),
                        pd.Series(surrogates, name="surrogates")], axis=1)

        if labels is not None:
            df = pd.concat([df, pd.Series(data=labels, name="labels")], axis=1)
            label_name = "labels"
            test_names = [Constants.true_positive_ratio, Constants.true_negative_ratio,
                          Constants.false_positive_ratio, Constants.false_negative_ratio,
                          Constants.prediction_ratio]
        else:
            label_name = None
            test_names = [Constants.prediction_ratio]
        # To specify likelihoods, user can provide either:
        # 1. An ndarray of likelihoods that gives likelihood of protected membership
        #   for each person. If this is given, we have to summarize the likelihoods at the surrogate level
        # 2. A dataframe ttehat has a row for each surrogate class value and
        #   a column for each likelihood value. The dataframe must have surrogate class as an index.
        if isinstance(memberships, pd.DataFrame):
            name = memberships.index.name
            likes_df = memberships.reset_index().rename(columns={name: 'surrogates'})
        else:
            if len(memberships) != df.shape[0]:
                len_predictions = len(predictions)
                len_likelihoods = len(memberships)
                raise InputShapeError("",
                                      "Likelihoods must either be a pandas dataframe with surrogates as index "
                                      "or be the same length as predictions vector"
                                      f"length of predictions {len_predictions}"
                                      f"length of likelihoods {len_likelihoods}")
            if isinstance(memberships, list) or isinstance(memberships, np.ndarray):
                interim_df = pd.DataFrame(data=memberships)
            elif isinstance(memberships, pd.Series):
                interim_df = pd.DataFrame(list(memberships.values))
            else:
                interim_df = memberships
            likes_detail = pd.concat([pd.Series(surrogates, name="surrogates"), interim_df], axis=1)
            likes_df = likes_detail.groupby(by="surrogates").mean()
            likes_df.columns = membership_names
            likes_df = likes_df.reset_index()
        summarizer = cls("surrogates", "surrogates", "predictions", true_name=label_name, test_names=test_names)
        return summarizer.make_summary_data(perf_df=df, surrogate_df=likes_df,warnings=warnings)

    def __init__(self, surrogate_surrogate_col_name: str,
                 surrogate_perf_col_name: str,
                 pred_name: str,
                 true_name: str = None,
                 max_shrinkage: float = 0.5,
                 test_names: str = None):
        """
        Initialize names of input variables needed to calculate summaries.
        Arguments:
            surrogate_surrogate_col_name: Name of surrogate variable in surrogate dataframe
            surrogate_perf_col_name: Name of surrogate variable in perfomance dataframe
            max_shrinkage: Not all columns in surrogate data frame will be in prediction dataframe or vice versa. Tells how many rows we can lose before throwing an error.
            test_names: Names of columns that will be come Ys in BiasCalculator. Usually confusion matrix columns.
        """
        self.surrogate_surrogate_col_name(surrogate_surrogate_col_name)
        self.surrogate_perf_col_name(surrogate_perf_col_name)
        self.pred_name(pred_name)
        self._true_name = None
        self.true_name(true_name)
        self.max_shrinkage(max_shrinkage)
        self._test_names = None
        self.test_names(test_names)

    def surrogate_surrogate_col_name(self, value: str = None):
        """
        Get or set surrogate column name in surrogate dataframe
        """
        if value:
            if self.col_name_checker(value):
                self._surrogate_surrogate_col_name = value
        return self._surrogate_surrogate_col_name

    def surrogate_perf_col_name(self, value: str = None):
        """
        Get or set surrogate column name in performance dataframe
        """
        if value:
            if self.col_name_checker(value):
                self._surrogate_perf_col_name = value
        return self._surrogate_perf_col_name

    def pred_name(self, value: str = None):
        """
        Get or set name of column that has predicted values
        """
        if value:
            if self.col_name_checker(value):
                self._pred_name = value
        return self._pred_name

    def true_name(self, value: str = None):
        """
        get or set column name that has true labels
        """
        if value:
            if self.col_name_checker(value):
                self._true_name = value
        return self._true_name

    def max_shrinkage(self, value: float = None):
        """
        Get or set how many columns we can lose without throwing an error
        """
        if value:
            if isinstance(value, float) and value < 1 and value > 0:
                self._max_shrinkage = value
            else:
                raise ValueError(f"Max shrinkage must be a float between 0 and 1. input value is: {value}.")
        return self._max_shrinkage

    @staticmethod
    def col_name_checker(value: str):
        """
        Helper function that makes sure column names have valid types
        """
        if not isinstance(value, str):
            raise ValueError(f"Column names must be strings. {value} is not a string.")
            # return False
        return True

    def check_performance_data(self, df: pd.DataFrame):
        """
        Checks specific to performance data.
        Arguments:
            df: dataframe with model performance data
        """
        if self.true_name() is not None:
            needed_names = [self.true_name()] + [self.pred_name()] + [self.surrogate_perf_col_name()]
        else:
            needed_names = [self.pred_name()] + [self.surrogate_perf_col_name()]
        return self.check_read_data(df, needed_names, "performance_data")

    @staticmethod
    def check_read_data(df: pd.DataFrame, needed_names: List[str], df_name: str, id_col_name: str = None):
        """
        Sanity check input data and raise errors when there are issues.
        """
        n_rows = df.shape[0]
        all_good = True
        if id_col_name:
            n_unique_ids = df[id_col_name].nunique()
            if not n_rows == n_unique_ids:
                raise Warning(f"Number of unique ids in {df_name} is: {n_unique_ids} but number of rows is {n_rows}")
        print(f"There are {n_rows} in {df_name}.")
        names = df.columns
        if not set(needed_names).issubset(set(names)):
            raise ValueError("Some necessary columns not in {0} data: {1} are missing.".format(df_name, list(
                set(needed_names) - set(names))))
        return all_good

    def test_names(self, value: List[str] = None) -> List[str]:
        """
        Get or set names for columns that have test statistics
        Arguments:

        """
        if value:
            if isinstance(value, list):
                self._test_names = value
            else:
                raise ValueError(f"Test names must be a list. {value} is not a list")
        if not self._test_names:
            if self.pred_name() and self.true_name():
                self._test_names = ["true_positive", "true_negative", "false_positive", "false_negative"]
            else:
                self._test_names = [Constants.prediction_ratio]
        return self._test_names

    def check_surrogate_data(self, df: pd.DataFrame) -> bool:
        """
        Checks specific to surrogate class data.
        """
        all_good = True
        needed_names = [self._surrogate_surrogate_col_name]
        if not self.check_read_data(df, needed_names, "surrogate data"):
            all_good = False
            return all_good
        else:
            all_good = (df[self._surrogate_surrogate_col_name].nunique() == df.shape[0])
            if not all_good:
                print("Input Surrogate data has duplicates. Surrogate data must be de-duplicated by surrogate column.")
        return all_good

    def check_merged_data(self, merged_df: pd.DataFrame,
                          performance_df: pd.DataFrame,
                          print_warnings: bool = True):
        """
        Make sure merged data hasn't lost too many rows due to inner join
        And make sure it hasn't increased in rows due to surrogate values duplicates
        Arguments:
        merged_df: data frame resulting from merge between surrogate_df and performance_df
        surrogate_df: surrogate data frame
        performance_df: Performance data frame
        """

        m_rows = float(merged_df.shape[0])
        p_rows = float(performance_df.shape[0])
        shrinkage = 1 - m_rows / p_rows

        if print_warnings:
            if shrinkage < 0:
                raise Warning(
                    f"Merged data has {m_rows}. Input performance data has {p_rows}. There may be duplicate surrogate calues in surrogate data.")
            elif shrinkage > self.max_shrinkage():
                print(f"Merged data has {m_rows}, but performance data only has {p_rows}.")
                raise ValueError(
                    "Merge between surrogate data and performance data results in loss of {0:.0}% of performance data.".format(
                        shrinkage))
            elif shrinkage > 0.2:
                warnings.warn(f"Merged data has {m_rows}, but performance data has {p_rows}.")

    def check_surrogate_confusion_matrix(self, confusion_df, merged_df):
        """
        Make sure confusion matrix is unique by surrogate class.
        Arguments:
            confusion_df: The summary by surrogate class
            merged_df: the original detail df.
        """
        n_rows = confusion_df.shape[0]
        n_unique_surrogates = merged_df[self._surrogate_surrogate_col_name].nunique()
        if not n_rows == n_unique_surrogates:
            # TODO This keeps printing during tests, can we turn off?
            print(
                f"Final dataframe has {n_rows} and {n_unique_surrogates} unique surrogate classes. There should be one row per surrogate class.")
            raise Warning("Possible missing surrogate classes in output data.")
            # return False
        return True

    def make_summary_data(self, perf_df: pd.DataFrame, surrogate_df: pd.DataFrame = None,warnings=True):
        """
        Function that merges two dfs to make a surrogate-based summary file that includes confusion matrix ratios.
        Arguments:
        surrogate_df: a dataframe unique by surrogate column that has class membership likelihoods for each surrogate class.
        perf_df: a dataframe that has surrogate class and performance columns. One row per scored individual.
        """
        self.check_performance_data(perf_df)
        self.check_surrogate_data(surrogate_df)
        merged_data = perf_df.merge(surrogate_df, left_on=self.surrogate_perf_col_name(),
                                    right_on=self.surrogate_surrogate_col_name())
        self.check_merged_data(merged_data, perf_df,warnings)

        # Create accuracy columns that measure true positive, true negative etc
        accuracy_df = pd.concat([merged_data[self.surrogate_surrogate_col_name()],
                                 self.confusion_matrix_actual(merged_data, self.pred_name(), self.true_name())], axis=1)
        # Use calc_accuracy_metrics to create surrogate-level summary
        #TODO: Accomodate cases where we don't have a binary classifier
        confusion_matrix_surrogate_summary = self.calc_accuracy_metrics(accuracy_df)
        self.check_surrogate_confusion_matrix(confusion_matrix_surrogate_summary, merged_data)
        return confusion_matrix_surrogate_summary.join(
            surrogate_df.set_index(surrogate_df[self.surrogate_surrogate_col_name()]))

    # Add columns to a pandas dataframe flagging each row as false positive, etc.
    def confusion_matrix_actual(self, test_df, pred_col, label_col):
        """
        Construct 0/1 variables for membership in a quadrant of the confusion matrix
        Arguments: test_df: dataframe with detail data that has a pred_column and a label_column
        pred_col: 0/1 predicted in class or not.
        label_col: 0/1 true label
        """
        # TODO: Replace this with confusion_matrix from utils.py
        if label_col is not None:
            correct = (test_df[pred_col] == test_df[label_col]).astype(int)
            correct.name = "correct"
            true_positive = (correct & (test_df[label_col] == 1)).astype(int)
            true_positive.name = "true_positive"
            true_negative = (correct & (test_df[label_col] == 0)).astype(int)
            true_negative.name = "true_negative"
            false_negative = ((correct == False) & (test_df[pred_col] == 0)).astype(int)
            false_negative.name = "false_negative"
            false_positive = ((correct == False) & (test_df[pred_col] == 1)).astype(int)
            false_positive.name = "false_positive"
            return pd.concat([true_positive, true_negative, false_positive, false_negative, test_df[pred_col], correct],
                             axis=1)
        else:
            # This means we only have predictions and no true labels
            return test_df[pred_col]

    def calc_accuracy_metrics(self, test_df):
        """
        Calculate TPR, etc from the confusion matrix columns
        Arguments:
            test_df: dataframe with detail data that will be rolled up by zip code
            group_col: surrogate column name
            acc_cols: accuracy columns that are in the dataframe as 0/1 and will be rolled up by zip
        """
        group_col = [self._surrogate_perf_col_name]

        if self.true_name() is not None:
            acc_cols = ["true_positive", "true_negative",
                        "false_positive", "false_negative"]
        else:
            acc_cols = []

        agg_dict = {}
        ac_cols = acc_cols + [self.pred_name()]
        for c in ac_cols:
            agg_dict[c] = "sum"
        agg_dict[group_col[0]] = Constants.weight_col_name
        check_accuracy = test_df \
            .groupby(group_col) \
            .agg(agg_dict) \
            .rename(columns={group_col[0]: Constants.weight_col_name})
        for c in ac_cols:
            check_accuracy[c + "_ratio"] = check_accuracy[c] / check_accuracy[Constants.weight_col_name]
        check_accuracy = check_accuracy.rename(
            columns={"_".join([self.pred_name(), "ratio"]): Constants.prediction_ratio})

        out_cols = [Constants.prediction_ratio, Constants.weight_col_name]

        if {Constants.true_negative_ratio, Constants.true_positive_ratio, Constants.false_negative_ratio,
            Constants.false_positive_ratio}.issubset(
            set(check_accuracy.columns)):
            out_cols = out_cols + [Constants.true_negative_ratio, Constants.true_positive_ratio,
                                   Constants.false_negative_ratio, Constants.false_positive_ratio]
            # Return a dataframe that has the stats by group. Use these to compare to expected values
        return check_accuracy[out_cols]
    #TODO: Needs string method