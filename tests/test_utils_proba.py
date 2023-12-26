import unittest
import numpy.random
import sklearn
import pandas as pd
import numpy as np
import inspect
from jurity.utils_proba import BiasCalculator, BiasCalcFromDataFrame, SummaryData
from jurity.utils_proba import unpack_bootstrap, check_memberships_proba_df
from jurity.utils import Constants
from jurity.utils_proba import get_bootstrap_results


class TestUtilsProba(unittest.TestCase):
    """
    These tests are for BiasCalculator
    """

    @classmethod
    def setUpClass(cls):
        # TODO Strip these names away from zip, race, etc. verbiage, let's keep it generic
        # TODO See Constants in utils to re-use string constants for column names,
        #  add the ones that are missing to Constants and use here
        # Define Constants needed for testing probabilistic utilities
        # Example summarized dataframe suitable for creating a bootstrap calculator
        cls.summarized_df = pd.DataFrame({"surrogate": [1, 2, 3, 4, 5],
                                          "W": [0.5, 0.1, 0.3, 0.7, 0.6],  # unprotected class
                                          "B": [0.2, 0.7, 0.5, 0.1, 0.3],  # protected class B
                                          "O": [0.3, 0.2, 0.2, 0.2, 0.1],  # protected class O
                                          "count": [30, 10, 6, 9, 2],  # weight column
                                          Constants.false_positive_ratio: [0.1, 0.5, 0.3, 0.1, 0.2],
                                          Constants.true_positive_ratio: [0.9, 0.5, 0.7, 0.9, 0.8],
                                          Constants.false_negative_ratio: [0.2, 0.4, 0.3, 0.2, 0.1],
                                          Constants.true_negative_ratio: [0.8, 0.6, 0.7, 0.8, 0.9]})
        cls.summarized_df[Constants.prediction_ratio] = \
            cls.summarized_df[Constants.true_positive_ratio] + cls.summarized_df[Constants.false_positive_ratio]
        # Create bias calculator from above dataframe
        cls.bcfd = BiasCalcFromDataFrame(["W", "B", "O"], "count", ["B", "O"],
                                         [Constants.false_positive_ratio,
                                          Constants.true_positive_ratio,
                                          Constants.false_negative_ratio,
                                          Constants.true_negative_ratio,
                                          Constants.prediction_ratio])
        # DataFrame matching the form that is returned bu BiasCalculator.run_bootstrap()
        # Tests transformations
        raw_boot_results = pd.DataFrame({
            Constants.false_negative_ratio: [0.25, 0.7, 0.5],
            Constants.false_positive_ratio: [0.3, 0.2, 0.5],
            Constants.true_negative_ratio: [0.2, 0.09, 0.2],
            Constants.true_positive_ratio: [0.25, 0.01, 0.25],
            Constants.prediction_ratio: [0.55, 0.21, 0.75],
            "class": ["W", "B", "O"]})
        # Add FRP, FNR, etc to match what's returned by the bootstrap
        cls.test_boot_results = cls.calc_rates(raw_boot_results)
        cls.bc = BiasCalculator.from_df(cls.summarized_df, membership_labels=[1, 2], membership_names=["W", "B", "O"],
                                        weight_warnings=False)

    @classmethod
    def calc_rates(cls, results):
        results[Constants.FPR] = results[Constants.false_positive_ratio] / (
                results[Constants.false_positive_ratio] + results[Constants.true_negative_ratio])
        results[Constants.FNR] = results[Constants.false_negative_ratio] / (
                results[Constants.false_negative_ratio] + results[Constants.true_positive_ratio])
        results[Constants.TPR] = results[Constants.true_positive_ratio] / (
                results[Constants.true_positive_ratio] + results[Constants.false_negative_ratio])
        results[Constants.TNR] = results[Constants.true_negative_ratio] / (
                results[Constants.true_negative_ratio] + results[Constants.false_positive_ratio])
        results[Constants.ACC] = results[Constants.true_positive_ratio] + results[Constants.true_negative_ratio]
        results[Constants.PRED_RATE] = results[Constants.prediction_ratio]
        return results

    def test_calc_one_bag_form(self):
        """
        The calling functions for calc_one_bag expects the output in a certain format.
        Make sure the return is as expected.
        --There is the right number of rows and columns
        --Multiple Y's are handled appropriately
        --Single Y's are handed appropriately
        """
        out = self.bc.calc_one_bag(self.summarized_df[["B", "O"]].to_numpy(),
                                   self.summarized_df[[Constants.false_positive_ratio, Constants.false_negative_ratio,
                                                       Constants.true_positive_ratio, Constants.true_negative_ratio,
                                                       Constants.prediction_ratio]].to_numpy(),
                                   self.summarized_df["count"].to_numpy())

        self.assertTrue(isinstance(out[Constants.false_positive_ratio], np.ndarray))
        self.assertTrue(isinstance(out[Constants.false_negative_ratio], np.ndarray))
        self.assertTrue(isinstance(out[Constants.true_positive_ratio], np.ndarray))
        self.assertTrue(isinstance(out[Constants.true_negative_ratio], np.ndarray))

    def test_calc_one_bag(self):
        """
        Test that calc_one_bag returns the (numerically) correct parameter estimates
        From an input X, Y, and W matrix
        --This function runs a single linear regression and outputs the parameter estimates
        --Create a test input X, Y, and W matrix.
        --Calculate the parameters that would be calculated using sklearn.linear_model.LinearRegression
        --Put the answers in the form that calc_one_boot should return
        --In this unit test, check whether the answers from calc_one_bag are equal to what comes out of sklearn.linear_model
        """
        x = np.array(self.summarized_df[["B", "O"]])
        y = np.array(self.summarized_df[self.bc.test_labels()])
        w = np.array(self.summarized_df["count"])

        out = self.bc.calc_one_bag(x, y, w)

        model = sklearn.linear_model.LinearRegression()
        pred_matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # Make sure each has the proper label
        for i, l in enumerate(self.bc.test_labels()):
            model.fit(x, np.array(self.summarized_df[l]), sample_weight=w)
            p = model.predict(pred_matrix)
            np.testing.assert_array_almost_equal(p, out[l])

    def test_run_boot_form(self):
        """
        Test that run_boot returns a dataframe that has expected results:
        --the correct number of parameter estimates
        --The correct number of samples
        --Do this by creating a BiasCalculator with a given test X,W, and Y
        --Make sure run_bootstrap gives an answer with the expected number of columns and rows.
        --Test different numbers of bootstraps and different numbers of columns for X and W
        """
        bootstrap = self.bc.run_bootstrap(5)
        self.assertEqual(bootstrap.shape, (15, 12))

    def test_transform_bootstrap_results_form(self):
        """
        Test that transform_results gives an output with the correct number of rows and columns
        """
        br = self.bc.transform_bootstrap_results(self.bc.run_bootstrap(5))
        self.assertEqual(br.shape, (3, 11), "Returned bootstrap has shape: {0}. Expected (3,11).".format(br.shape))
        test_cols = [s in br.columns for s in
                     [Constants.FPR, Constants.FNR, Constants.TPR, Constants.TNR, Constants.ACC, Constants.PRED_RATE]]
        self.assertTrue(np.all(test_cols), "Not all tests are returned by bootstrap transform")

    def test_transform_bootstrap_results_answer(self):
        """
        Test that transform_bootstrap_results gives the correct numbers
        """
        test_these = [Constants.FNR, Constants.FPR, Constants.TNR, Constants.TPR, Constants.ACC,
                      Constants.PRED_RATE, Constants.ACC]
        boot = self.bc.transform_bootstrap_results(self.test_boot_results)
        ratios_added = boot.groupby("class").mean()
        np.testing.assert_array_almost_equal(np.array(boot[test_these]), np.array(ratios_added[test_these]))

    """
    These tests are for MakeBiasCalcFromDataFrame 
    """

    def test_make_Y_matrix(self):
        """
        Test that make_x_matrix creates a two-dimensional numpy array with the correct numbers
        """
        Y = self.bcfd.get_Y_matrix(self.summarized_df)
        ans = [[0.1, 0.9, 0.2, 0.8, 1.0],
               [0.5, 0.5, 0.4, 0.6, 1.0],
               [0.3, 0.7, 0.3, 0.7, 1.0],
               [0.1, 0.9, 0.2, 0.8, 1.0],
               [0.2, 0.8, 0.1, 0.9, 1.0]]
        np.testing.assert_array_almost_equal(Y, ans)

    def test_make_X_matrix(self):
        """
        test that make_X_matrix gives a two-dimensional array with the correct numbers
        """
        X = self.bcfd.get_X_matrix(self.summarized_df)
        ans = [[0.2, 0.3],
               [0.7, 0.2],
               [0.5, 0.2],
               [0.1, 0.2],
               [0.3, 0.1]]
        np.testing.assert_array_almost_equal(X, ans)

    def test_make_W_array(self):
        """
        Test that make_W_array gives a one-dimensional array with the correct numbers
        """
        w = self.bcfd.get_W_array(self.summarized_df)
        ans = [30, 10, 6, 9, 2]
        np.testing.assert_array_equal(w, ans)

    def test_make_bias_calculator_filter(self):
        """
        Test that make_bias_calculator filters rows with small counts
        """
        bc_filtered = self.bcfd.get_bias_calculator(self.summarized_df, 7,weight_warnings=False)
        self.assertEqual(bc_filtered.X().shape[0], 3)

    def test_make_bias_calculator_names(self):
        """
        Test to make certain make_bias_calculator has the correct class labels and omitted category
        """
        self.assertTrue("B" in self.bc.class_labels()[1])
        self.assertTrue("O" in self.bc.class_labels()[1])
        self.assertFalse("W" in self.bc.class_labels()[1])
        self.assertEqual("W", self.bc.class_labels()[0][0])

    def test_bias_maker_bad_data(self):
        # not list
        self.assertRaises(ValueError, BiasCalcFromDataFrame, "B", "count", [1],
                          [Constants.false_positive_ratio, Constants.true_positive_ratio,
                           Constants.false_negative_ratio,
                           Constants.true_negative_ratio])
        self.assertRaises(ValueError, BiasCalcFromDataFrame, ["W", "B", "O"], "count", [1, 2],
                          Constants.false_positive_ratio)
        # not string
        self.assertRaises(ValueError, BiasCalcFromDataFrame, ["W", "B", "O", 0], "count", [1, 2],
                          [Constants.false_positive_ratio, Constants.true_positive_ratio,
                           Constants.false_negative_ratio,
                           Constants.true_negative_ratio])
        self.assertRaises(ValueError, BiasCalcFromDataFrame, ["W", "B", "O"], "count", [1, 2],
                          [Constants.false_positive_ratio, Constants.true_positive_ratio,
                           Constants.false_negative_ratio, Constants.true_negative_ratio,
                           0])
        # column missing
        fac = BiasCalcFromDataFrame(["W", "B", "O", "hello world"], "N", [1, 2],
                                    [Constants.false_positive_ratio, Constants.true_positive_ratio,
                                     Constants.false_negative_ratio,
                                     Constants.true_negative_ratio])
        self.assertRaises(ValueError, fac.get_bias_calculator, self.summarized_df, 1,weight_warnings=False)
        fac = BiasCalcFromDataFrame(["W", "B", "O"], "N", [1, 2],
                                    [Constants.false_positive_ratio, Constants.true_positive_ratio,
                                     Constants.false_negative_ratio,
                                     Constants.true_negative_ratio, "hello world"])
        self.assertRaises(ValueError, fac.get_bias_calculator, self.summarized_df, 1, weight_warnings=False)
        fac = BiasCalcFromDataFrame(["W", "B", "O"], "hello world", [1, 2],
                                    [Constants.false_positive_ratio, Constants.true_positive_ratio,
                                     Constants.false_negative_ratio,
                                     Constants.true_negative_ratio])
        self.assertRaises(ValueError, fac.get_bias_calculator, self.summarized_df, 1,weight_warnings=False)

    def test_summary(self):
        predictions = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        surrogates = [1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 5]
        labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
        memberships = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
                                [0.2, 0.8],
                                [0.1, 0.9], [0.1, 0.9],
                                [0.25, 0.75],
                                [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])

        summarize = SummaryData("surrogates", "surrogates", "predictions", "labels")

        test_df = pd.DataFrame({"prediction": predictions, "surrogates": surrogates, "labels": labels})
        # use other confusion matrix to test
        with_summary = summarize.confusion_matrix_actual(test_df, "prediction", "labels")

        result = pd.concat([pd.Series(surrogates, name="surrogates"), with_summary], axis=1).groupby(
            "surrogates").mean()

        result.columns = ["_".join([c, "ratio"]) for c in result.columns]

        df = SummaryData.summarize(predictions, memberships, surrogates, labels)

        self.assertEqual(df.shape[0], len(set(surrogates)),
                         "MakeSummaryData.summarize_detail returns wrong number of rows.")

        self.assertTrue(
            df[[Constants.true_positive_ratio, Constants.true_negative_ratio, Constants.false_positive_ratio,
                Constants.false_negative_ratio, "prediction_ratio"]].equals(result.drop("correct_ratio", axis=1)))

    def test_bias_calc_from_dataframe(self):
        predictions = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        memberships = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
                                [0.2, 0.8],
                                [0.1, 0.9], [0.1, 0.9],
                                [0.25, 0.75],
                                [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])
        surrogates = [1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 5]
        labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]

        df = SummaryData.summarize(predictions, memberships, surrogates, labels)

        bc = BiasCalculator.from_df(df, [1], ["A", "B"], weight_warnings=False)

        self.assertTrue(np.all(np.isclose(bc.X().ravel(), df["B"].values)),
                        "X matrix in bias calculator does not match surrogate class probabilities."
                        "\nBias calc has:{0},original df has: {1}.\nCompare result is: {2}"
                        .format(bc.X().ravel(), df["B"].values, bc.X().ravel() == (df["B"].values)))

        self.assertEqual(bc.Y().shape[0], len(np.unique(surrogates)),
                         "Y matrix in BiasCalculator has wrong length. Length is {0}. Should be {1}.".format(
                             bc.Y().shape[0], len(np.unique(surrogates))))

    def test_unpack_bootstrap(self):
        """
        Make sure unpack bootstrap can return all expected results
        """
        c = pd.Series(["W", "NW"], name="class")
        answer_dict = {"FPR": [0.6, 0.689655],
                       "FNR": [0.5, 0.985915],
                       "TPR": [0.5, 0.014085],
                       "TNR": [0.4, 0.310345],
                       "ACC": [0.45, 0.10]}
        raw_boot_results = pd.DataFrame({
            "false_negative_ratio": [0.25, 0.7],
            "false_positive_ratio": [0.3, 0.2],
            "true_negative_ratio": [0.2, 0.09],
            "true_positive_ratio": [0.25, 0.01],
            "prediction_ratio": [0.55, 0.21]})
        test_boot_results = pd.concat([pd.DataFrame.from_dict(answer_dict), raw_boot_results], axis=1).set_index(c)

        for label, answer in answer_dict.items():
            self.assertEqual(unpack_bootstrap(test_boot_results, label, [1]),
                             (answer[1], answer[0]),
                             f"unpack bootstrap returns unexpected answer for {label}\n" +
                             "expected {0}, got {1} instead.".format(unpack_bootstrap(test_boot_results, label, [1]),
                                                                     (answer[1], answer[0])))

    def test_unpack_bootstrap_err(self):
        test_unpack = self.bc.transform_bootstrap_results(self.test_boot_results)
        self.assertRaises(ValueError, unpack_bootstrap, test_unpack, "FNR", [1, 2])

    def test_from_df(self):
        self.assertRaises(ValueError, BiasCalculator.from_df, self.summarized_df,
                          [3], ["W", "B", "O"], weight_warnings=False)

    def test_summarizer(self):
        predictions = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        surrogates = [1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 5]
        labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
        memberships = pd.DataFrame(np.array([[0.5, 0.5], [0.2, 0.8],
                                             [0.1, 0.9],
                                             [0.25, 0.75],
                                             [0.3, 0.7]]))
        memberships.columns = ["C", "D"]
        memberships["s"] = pd.Series([1, 2, 3, 4, 5])
        summary = SummaryData.summarize(predictions, memberships.set_index("s"), surrogates, labels)
        self.assertTrue(summary.shape[0] == 5, "Summarizer returns dataframe with wrong shape")
        self.assertTrue(np.all(~summary["C"].apply(np.isnan)), "Summarizer inserts NaN values.")
        self.assertTrue(np.all(~summary["D"].apply(np.isnan)), "Summarizer inserts NaN values.")
        expected_cols={'prediction_ratio', 'count', 'true_negative_ratio',
                        'true_positive_ratio', 'false_negative_ratio', 'false_positive_ratio',
                        'surrogates', 'C', 'D'}
        returned_cols=set(summary.columns)
        self.assertTrue(expected_cols==returned_cols,
                    f"Summary dataframe does not return correct columns. \nReturns: {returned_cols}. \nExpected: {expected_cols}")

    # TODO: Write tests for check_memberships_proba


class UtilsProbaSimulator:
    """
    Simulation functions used to test probabilistic fairness.
    Can be used by other researchers to simulate different levels of unfairness and test their own methods.
    Members:
    _rates_dict: Dictionary of dictionaries with expected fairness metrics for each protected class,
        has the form: {"class1":{'pct_positive':float,'fpr':float,'fnr':float}, "class2":{'pct_positive'}...}
    _surrogate_name: Name of surrogate column for input dataframe
    rng: numpy random number generators. Set on initialization if you want to set the seed for your simulation
    """
    def __init__(self, model_rates_dict: dict,
                 in_rng: numpy.random.Generator = None,
                 surrogate_name: str ="surrogate"):
        self.rates_dict(model_rates_dict)
        self.surrogate_name(surrogate_name)
        if in_rng is not None:
            self.rng(in_rng)
        else:
            self._rng = numpy.random.default_rng()

    def rates_dict(self, v=None):
        """
        Set and get rates_dict dictionary
        """
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Rates must be a dictionary. Input rates has type: {0}".format(type(dict)))
            for k,value in v.items():
                if not set(value.keys()) == {"pct_positive", "fpr", "fnr"}:
                    raise ValueError("Rates must have ".format("pct_positive", "fpr", "fnr"))
                if not isinstance(k,str):
                    raise ValueError("Keys for main dictionary must be strings")
                for k2,value2 in value.items():
                    if not isinstance(value2, float):
                        raise ValueError("Rates must be floats.")
            self._rates_dict = v
        return self._rates_dict

    def surrogate_name(self,v=None):
        if not v is None:
            if not isinstance(v,str):
                raise ValueError("surrogate_name must be a string.\n {0} supplied intead.".format(type(v)))
            self._surrogate_name=v
        return self._surrogate_name
    def rng(self, v=None):
        """
        Set and get random number generator
        """
        if v is not None:
            if not isinstance(v, numpy.random.Generator):
                raise ValueError("rng argument must be a numpy random number generator.")
            self._rng = v
        return self._rng

    # For the simulation, build "True" protected groups based on population
    def assign_protected(self, population_data, generator,
                         membership_values=None):
        # Passing in the global random number generator
        # Lets us make sure that we're not accidentally resetting the seed
        if membership_values is None:
            membership_values=["W", "O", "B", "T", "A", "AI"]
        surrogate_protected_prob_grouped = population_data.groupby(self.surrogate_name())
        surrogate_groups = []
        for name, group in surrogate_protected_prob_grouped:
            probs = [group[v].unique()[0] for v in membership_values]
            group["class"] = generator.choice(list(membership_values), len(group), p=probs)
            surrogate_groups.append(group)
        out_data = pd.concat(surrogate_groups)
        return out_data

    def assign_protected_and_accuracy(self, input_data, rates_by_protected, generator,
                                      protected_name="class"):
        # Assign everyone a "true" race for simulation purposes
        if not set(rates_by_protected.keys()).issubset(set(input_data.columns)):
            raise ValueError("Input dataframe does not have same column names as keys in rates.")
        protected_assignments = self.assign_protected(input_data, generator,
                                                      membership_values=list(rates_by_protected.keys()))
        model_outcomes = self.model_outcome_by_protected(protected_assignments, rates_by_protected,
                                                         protected_col_name=protected_name)
        return model_outcomes

    def confusion_matrix_prob(self, percent_positive, fpr, fnr, verbose=False):
        """
        Calculate the probability of any given individual falling into each quadrant of the confusion matrix
        percent_positive: Percent of positive cases in the training data
        fpr: False Positive Rate from the hypothetical model
        fnr: False Negative Rate from the hypothetical model
        """
        fp_ratio = (1 - percent_positive) * fpr
        fn_ratio = percent_positive * fnr
        tn_ratio = (1 - percent_positive) - fp_ratio
        tp_ratio = percent_positive - fn_ratio
        probs = [fp_ratio, fn_ratio, tn_ratio, tp_ratio]
        if verbose:
            print("Expected FPR: " + str(fpr))
            print("Expected FNR: " + str(fnr))
            print("Expected TPR: " + str(tp_ratio / (tp_ratio + fn_ratio)))
            print("Expected TNR: " + str(tn_ratio / (tn_ratio + fp_ratio)))
            print("Expected Accuracy: " + str((tn_ratio + tp_ratio)))
        # print("% of FP, FN, TN, TP among total: ")
        return probs

    def model_outcome_by_protected(self, protected_assignment, rates_by_protected,
                                   protected_col_name="class"):
        """
        Assing each individual into a column of the confusion matrix based on probabilities for their class.
        """
        protected_prob_grouped = protected_assignment.groupby(protected_col_name)

        classified_groups = []
        for name, group in protected_prob_grouped:
            rates_dict = rates_by_protected[name]
            probs = self.confusion_matrix_prob(rates_dict["pct_positive"], rates_dict["fpr"], rates_dict["fnr"])
            group['pred_category'] = np.random.choice(['fp', 'fn', 'tn', 'tp'], len(group), p=probs)
            group['label'] = np.where(group['pred_category'].isin(['tp', 'fn']), 1, 0)
            group['prediction'] = np.where(group['pred_category'].isin(['tp', 'fp']), 1, 0)
            classified_groups.append(group)
        classified_data = pd.concat(classified_groups)
        return classified_data

    # Add columns to a pandas dataframe flagging each row as false positive, etc.
    def accuracy_columns(self, test_data: pd.DataFrame, pred_col: str, label_col:str)->pd.DataFrame:
        """
        Add indicators for each confusion matrix quadrant. Simplifies calculating rates.
        test_data: Input dataframe
        pred_col: Name of column with predicted class
        label_col: Name of column with actual class
        """
        correct=(test_data[pred_col] == test_data[label_col]).astype(int)
        correct.name="correct"
        true_positive=(correct & (test_data[label_col] == 1)).astype(int)
        true_positive.name="true_positive"
        true_negative=(correct & (test_data[label_col] == 0)).astype(int)
        true_negative.name="true_negative"
        false_negative=(~(correct) & (test_data[pred_col] == 0)).astype(int)
        false_negative.name="false_negative"
        false_positive = (~(correct) & (test_data[pred_col] == 1)).astype(int)
        false_positive.name="false_positive"
        return pd.concat([test_data,correct,true_positive,true_negative,false_negative,false_positive],axis=1)

    def explode_dataframe(self, df, count_name="count",surrogate_name="surrogate"):
        """
        Given a dataframe that has a count, produce a number of identical rows equal to that count
        df: pd.DataFrame with columns: count, class_1, class_2, ...  Class names must match keys from
        self._rates_dict
        count_name; name of count variable.
        """
        names=list(self.rates_dict().keys())
        if not set(names).issubset(df.columns):
            raise ValueError(f"DataFrame column names do not match keys in rates dictionary. Rates dict has: {names}.")
        check_memberships_proba_df(df[list(self.rates_dict().keys())],set(df.index.values),names)
        e_df = df.loc[df.index.repeat(df[count_name])].drop("count", axis=1)
        return self.assign_protected_and_accuracy(e_df, self._rates_dict, self._rng)


# Simulations to ensure numbers accuracy
class TestWithSimulation(unittest.TestCase):
    """
    Simulation tests for whether numbers are correct based on simulated inputs
    """
    @classmethod
    def setUpClass(cls) -> None:
        input_df = pd.DataFrame({"surrogate": list(range(0, 99)),
                                 "count": [473, 516, 529, 497, 476, 529, 493, 497, 503, 490, 507, 514, 524,
                                           485, 470, 513, 501, 505, 488, 510, 518, 501, 506, 484, 493, 504,
                                           477, 537, 491, 535, 517, 472, 510, 478, 518, 449, 503, 503, 509,
                                           537, 504, 533, 493, 482, 495, 497, 495, 465, 501, 512, 468, 470,
                                           549, 510, 503, 524, 496, 526, 481, 478, 557, 487, 511, 493, 486,
                                           517, 497, 517, 504, 472, 500, 493, 494, 504, 464, 543, 513, 486,
                                           488, 485, 486, 480, 519, 494, 509, 501, 494, 515, 522, 500, 532,
                                           512, 490, 486, 516, 495, 530, 542, 588],
                                 "W": [0.36137754, 0.83653862, 0.98303716, 0.52943704, 0.80254777,
                                       0.86131181, 0.78572192, 0.79557292, 0.94314381, 0.98431145,
                                       0.97623762, 0.93004508, 0.94375, 0.87960053, 0.9400488,
                                       0.65223399, 0.97161572, 0.93023638, 0.95479798,
                                       0.89171974, 0.97123894, 0.95230717, 0.22796454, 0.66407934,
                                       0.97238007, 0.1840455, 0.98376723, 0.98458971, 0.98717949,
                                       0.79029126, 0.91857947, 0.88296571, 0.88260341, 0.82066313,
                                       0.64678885, 0.9261571, 0.45299573, 0.65396544, 0.75473463,
                                       0.71911322, 0.91455557, 0.55873562, 0.68013085, 0.92248062,
                                       0.90759089, 0.94984894, 0.97193437, 0.86554149, 0.8591954,
                                       0.84304381, 0.94390104, 0.79376303, 0.79835391, 0.99337748,
                                       0.97175349, 0.88572513, 0.64860465, 0.83355958, 0.97107438,
                                       0.77651098, 0.712, 0.98913043, 0.49761412, 0.46572154,
                                       0.57169428, 0.70574713, 0.96366279, 0.95924765, 0.94465361,
                                       0.96124031, 0.89406286, 0.96956522, 0.96589359, 0.62571886,
                                       0.75265434, 0.63236253, 0.51799237, 0.98781179, 0.7579503,
                                       0.95535714, 0.96797005, 0.55753216, 0.45074149, 0.7496386,
                                       0.76968338, 0.8861327, 0.44629554, 0.67914006, 0.96602972,
                                       0.98126951, 0.31041469, 0.64384805, 0.26402656, 0.9602649,
                                       0.7840254, 0.25, 0.47070506, 0.95252806, 0.97494781],
                                 "B": [0.53346927, 0.10651034, 0.00565428, 0.31113021, 0.00636943,
                                       0.00309656, 0.02780789, 0.17230903, 0., 0.0040674,
                                       0.0019802, 0.01768377, 0., 0.02423198, 0.0232369,
                                       0.30792262, 0.00873362, 0.00246225, 0.00816498,
                                       0.10191083, 0., 0.00211297, 0.00993179, 0.30616645,
                                       0.00308356, 0.54513636, 0.00245023, 0.00176678, 0.003663,
                                       0.00388349, 0.04850585, 0.01071464, 0.00364964, 0.05440557,
                                       0.062256, 0.00604955, 0.28707358, 0.32713041, 0.0746768,
                                       0.20776774, 0.00648778, 0.04176577, 0.05834996, 0.,
                                       0.05076383, 0.00634441, 0.00345423, 0.1021097, 0.10057471,
                                       0.0031053, 0.037687, 0.01223824, 0.03703704, 0.,
                                       0.00110051, 0.00550493, 0.14318182, 0.03979129, 0.00295159,
                                       0.02460712, 0.131, 0.0013587, 0.4413433, 0.03242778,
                                       0.08792695, 0.00413793, 0.00872093, 0., 0.03087349,
                                       0.00775194, 0.0007761, 0.00124224, 0., 0.02630338,
                                       0.0307324, 0.05933493, 0.22485763, 0.00141723, 0.10647612,
                                       0.00714286, 0.00207987, 0.18627493, 0.24671719, 0.07806288,
                                       0.18856823, 0.03197848, 0.03738318, 0.28863346, 0.00212314,
                                       0.00208116, 0.66961591, 0.14106055, 0.02829685, 0.00331126,
                                       0.17769831, 0.73891626, 0.44502766, 0.0205952, 0.00991649],
                                 "A": [5.91596800e-03, 2.47713860e-02, 8.07754000e-04, 2.57842700e-03,
                                       5.73248410e-02, 2.81505000e-04, 2.94265860e-02, 1.30208300e-03,
                                       0.00000000e+00, 5.81058000e-04, 0.00000000e+00, 2.40984740e-02,
                                       0.00000000e+00, 1.73882190e-02, 1.32450330e-02, 4.26070900e-03,
                                       4.36681200e-03, 1.05055810e-02, 3.87205400e-03,
                                       0.00000000e+00, 2.21238900e-03, 7.28472800e-03, 5.78243384e-01,
                                       3.44976300e-03, 6.21118000e-03, 1.47549949e-01, 2.45023000e-03,
                                       2.35571300e-03, 1.83150200e-03, 5.82524300e-03, 6.35195600e-03,
                                       2.58275740e-02, 9.73236000e-03, 3.11412300e-03, 2.04738490e-02,
                                       8.06606500e-03, 1.65801967e-01, 3.69221700e-03, 9.25238150e-02,
                                       1.78298090e-02, 4.81066290e-02, 1.92019670e-01, 1.76720338e-01,
                                       5.81395300e-03, 7.06588200e-03, 2.38670690e-02, 6.47668400e-03,
                                       3.65682100e-03, 2.01149430e-02, 1.07609500e-03, 2.01380900e-03,
                                       2.88278490e-02, 1.34430727e-01, 0.00000000e+00, 1.10051400e-03,
                                       1.25284740e-02, 8.30549680e-02, 1.63534550e-02, 1.29870130e-02,
                                       7.54941230e-02, 6.80000000e-02, 1.35869600e-03, 1.81867290e-02,
                                       2.24096189e-01, 2.89820764e-01, 9.19540000e-04, 0.00000000e+00,
                                       0.00000000e+00, 2.25903600e-03, 0.00000000e+00, 7.76096000e-04,
                                       3.72670800e-03, 0.00000000e+00, 5.37381000e-03, 1.09609668e-01,
                                       1.76881837e-01, 1.22097753e-01, 4.25170100e-03, 8.01628690e-02,
                                       0.00000000e+00, 4.15973400e-03, 5.07935840e-02, 1.42287842e-01,
                                       8.31225200e-03, 7.28495400e-03, 1.79318600e-03, 4.84085060e-01,
                                       5.32141300e-03, 0.00000000e+00, 1.04058300e-03, 8.02083800e-03,
                                       1.72436150e-02, 6.05144038e-01, 0.00000000e+00, 5.80164900e-03,
                                       1.23152700e-03, 3.12101010e-02, 4.73689600e-03, 9.91649300e-03],
                                 "AI": [0.00289505, 0.00327419, 0.00080775, 0.00673256, 0.01910828,
                                        0.00487942, 0.01291899, 0.00390625, 0.02341137, 0.00232423,
                                        0.00693069, 0.00303398, 0.03125, 0.00638751, 0.00209132,
                                        0.00644864, 0.00218341, 0.02462246, 0.01010101,
                                        0., 0.00221239, 0.01130944, 0.00206634, 0.0064683,
                                        0.00519801, 0.01611492, 0.00398162, 0.0010797, 0.0018315,
                                        0.14368932, 0.00382561, 0.0088958, 0.0620438, 0.00604506,
                                        0.01621219, 0.00384098, 0.00312494, 0.003003, 0.00368564,
                                        0.00339776, 0.00123577, 0.01097577, 0.00234328, 0.00387597,
                                        0.00404611, 0., 0.00647668, 0.00506329, 0.,
                                        0.003505, 0.00143844, 0.00444203, 0.00068587, 0.,
                                        0.00696992, 0.00949127, 0.00445032, 0.00566326, 0.00118064,
                                        0.00356561, 0.004, 0.0013587, 0.00252093, 0.00836883,
                                        0.00185999, 0.18528736, 0.00290698, 0.02507837, 0.00414157,
                                        0., 0.0007761, 0.00248447, 0.01227831, 0.03516546,
                                        0.00847458, 0.00365862, 0.0080731, 0.00113379, 0.00206,
                                        0.00267857, 0., 0.00719417, 0.00434442, 0.0084026,
                                        0.00168114, 0.01165571, 0.00040634, 0.00229885, 0.00424628,
                                        0.0062435, 0.00124034, 0.00570465, 0.00241417, 0.01655629,
                                        0.00229877, 0.00246305, 0.00305008, 0.00308928, 0.00156576],
                                 "O": [0.08065856, 0.00486167, 0.00323102, 0.12304827, 0.02547771,
                                       0.11663695, 0.11456059, 0.01041667, 0., 0.00348635,
                                       0.0029703, 0.00684813, 0., 0.05125216, 0.00464738,
                                       0.0040304, 0.0014556, 0.00902823, 0.00841751,
                                       0., 0.00884956, 0.00444731, 0.00890973, 0.00043122,
                                       0.00110127, 0.0441155, 0., 0.00284649, 0.,
                                       0.00194175, 0.00512487, 0.03941929, 0.00121655, 0.08976003,
                                       0.20229474, 0.04321106, 0.05801166, 0.00379068, 0.0443553,
                                       0.02915006, 0.01694766, 0.12696857, 0.04009095, 0.05813953,
                                       0.00588166, 0.00966767, 0.00043178, 0.00421941, 0.00862069,
                                       0.13346656, 0.00575374, 0.14241682, 0.00205761, 0.,
                                       0.00403522, 0.02923311, 0.07679704, 0.07582828, 0.00354191,
                                       0.08619096, 0.057, 0.0013587, 0.02755019, 0.22448029,
                                       0.02908353, 0.00367816, 0.00290698, 0.01253918, 0.0060241,
                                       0.01550388, 0.08731083, 0.01428571, 0.01500682, 0.27840106,
                                       0.04434192, 0.09237122, 0.0574504, 0.00113379, 0.02941934,
                                       0.01785714, 0.00540765, 0.15478587, 0.11465343, 0.13633899,
                                       0.02129448, 0.04542738, 0.01029392, 0.01217539, 0.00424628,
                                       0.00104058, 0.00372101, 0.1730844, 0.06462546, 0.,
                                       0.01091002, 0.00738916, 0.02291105, 0.00926784, 0.00104384],
                                 "T": [0.01568361, 0.02404379, 0.00646204, 0.02707349, 0.08917198,
                                       0.01379375, 0.02956402, 0.01649306, 0.03344482, 0.00522952,
                                       0.01188119, 0.01829057, 0.025, 0.02113961, 0.01673057,
                                       0.02510364, 0.01164483, 0.02314511, 0.01464646,
                                       0.00636943, 0.01548673, 0.02253839, 0.17288422, 0.01940492,
                                       0.0120259, 0.06303777, 0.00735069, 0.0073616, 0.00549451,
                                       0.05436893, 0.01761224, 0.03217699, 0.04075426, 0.02601209,
                                       0.05197437, 0.01267524, 0.03299213, 0.00841825, 0.03002381,
                                       0.02274142, 0.01266661, 0.06953461, 0.04236462, 0.00968992,
                                       0.02465164, 0.0102719, 0.01122625, 0.01940928, 0.01149425,
                                       0.01580323, 0.00920598, 0.01831203, 0.02743484, 0.00662252,
                                       0.01504035, 0.05751708, 0.04391121, 0.02880414, 0.00826446,
                                       0.0336312, 0.028, 0.00543478, 0.01278473, 0.04490537,
                                       0.01961447, 0.10022989, 0.02180233, 0.0031348, 0.01204819,
                                       0.01550388, 0.01629802, 0.00869565, 0.00682128, 0.02903743,
                                       0.0541871, 0.03539086, 0.06952876, 0.0042517, 0.02393138,
                                       0.01696429, 0.0203827, 0.0434193, 0.04125563, 0.01924467,
                                       0.01148781, 0.02301255, 0.02153596, 0.01243082, 0.02335457,
                                       0.00832466, 0.00698722, 0.01905873, 0.03549293, 0.01986755,
                                       0.01926585, 0., 0.02709604, 0.00978272, 0.0026096]})

        # Small groups are too unstable for unit tests. Summarize to W,B,O
        input_df["O"] = input_df["O"] + input_df['A'] + input_df['AI'] + input_df['T']
        input_df = input_df.drop(["A", "AI", "T"], axis=1)
        # Numerical errors can make it so np.choice fails, make sure all categories sum to 1.
        input_df["W"] = 1 - (input_df['B'] + input_df['O'])

        cls.rates_dict = {"W": {"pct_positive": 0.1, "fpr": 0.1, "fnr": 0.1},
                          "B": {"pct_positive": 0.2, "fpr": 0.2, "fnr": 0.35},
                          'O': {"pct_positive": 0.1, "fpr": 0.1, "fnr": 0.1}}

        cls.rng = np.random.default_rng(347123)
        cls.sim=UtilsProbaSimulator(cls.rates_dict,in_rng=cls.rng)
        cls.surrogate_df = input_df[["surrogate", "W", "B", "O"]]
        cls.test_data = cls.sim.explode_dataframe(input_df[["surrogate", "count","W", "B", "O"]].set_index("surrogate")).reset_index()
        summary_df = SummaryData.summarize(cls.test_data["prediction"], cls.surrogate_df.set_index("surrogate"),
                                           cls.test_data["surrogate"], cls.test_data["label"])

        cls.bc = BiasCalculator.from_df(summary_df, [1, 2], ["W", "B", "O"])
    def test_membership_as_df(self):
        """
        Check output from get_bootstrap_results when inputs are a surrogate dataframe
        """
        results = get_bootstrap_results(self.test_data["prediction"], self.surrogate_df.set_index("surrogate"),
                                        self.test_data["surrogate"], [1, 2], self.test_data["label"])

        self.assertTrue(isinstance(results, pd.DataFrame), "get_bootstrap_results does not return a Pandas DataFrame.")
        self.assertTrue(
            {Constants.FPR, Constants.FNR, Constants.TNR, Constants.TPR, Constants.ACC}.issubset(set(results.columns)),
            "get_bootstrap_results does not return a dataframe with all expected binary metrics. Columns in DataFrame are:{0}".format(
                results.columns))
        self.assertTrue(set(results.index.values) == set(self.surrogate_df.drop(["surrogate"], axis=1).columns),
                        "get_bootstrap_results does not return a dataframe with rows index equal to the columns of input surrogate dataframe.\n"
                        "returned rows are:{0}\n"
                        "returned columns are: {1}\n".format(results.index.values, self.surrogate_df.columns))

    @classmethod
    def boot_stats(self, n_boots, n_loops):
        """
        Run bootstrap a specified number of times and return mean and std_dev
        """
        all_boots = []
        for i in range(0, n_loops):
            b = self.bc.run_bootstrap(n_boots)
            all_boots.append(self.bc.transform_bootstrap_results(b))
        summary = pd.concat(all_boots, axis=0).groupby('class')
        return summary.mean(), summary.std()

    def test_bootstrap_ranges(self):
        """
        Test whether bootstrap returns values that are expected based on simulated data
        """
        # Need to build a confidence interval where we expect values to be.
        # This requires calculation of theoretical variance/covariance matrix based on linear regression
        n_row = self.bc.X().shape[0]
        x = np.hstack((np.ones((n_row, 1)), self.bc.X()))
        # The variance-covariance matrix of a linear estimator based on input X is:
        invxTx = np.linalg.inv(np.dot(x.T, x))

        pred_matrix=np.array([[1.0,0.0,0.0],[1.0,1.0,0.0],[1.0,0.0,1.0]])
        # The variance-covariance matrix  of a linear calculation based on a prediction matrix is as follows.
        # Only need the diagonal for this calculation
        x_portion_variance = pd.Series(np.diag(np.dot(np.dot(pred_matrix, invxTx), pred_matrix.T)),
                                       index = self.bc.all_class_labels())
        x_portion_variance.name = 'x_var'

        # Get confusion matrix probabilities and variances from input rates_dict.
        in_vars_dict = {}
        in_means_dict = {}
        for k, v in self.rates_dict.items():
            a = self.sim.confusion_matrix_prob(v['pct_positive'], v['fpr'], v['fnr'])
            # These are based on variance of a proportion: p(1-p)
            in_vars_dict[k] = [r * (1 - r) for r in a]
            in_means_dict[k] = a

        df_vars = pd.DataFrame.from_dict(in_vars_dict)
        df_vars['stat_name'] = ['false_positive_var', 'false_negative_var', 'true_negative_var', 'true_positive_var']
        var_y = df_vars.set_index('stat_name').T

        df_in = pd.DataFrame.from_dict(in_means_dict)
        df_in['stat_name'] = ['false_positive_in', 'false_negative_in', 'true_negative_in', 'true_positive_in']
        in_y = df_in.set_index('stat_name').T

        n_boots = 100
        n_loops = 5
        mean, std = self.boot_stats(n_boots, n_loops)
        names = ['false_positive', 'false_negative', 'true_positive', 'true_negative']
        ratio_names = [n + "_ratio" for n in names]
        var_components = pd.concat([mean[ratio_names], in_y, var_y, x_portion_variance], axis=1)

        z = 1.65  # from z-table, predictions are N(in_y,var_pred_y)
        for n in names:
            # Prediction variance is sigma_y*pred_matrix*inv(X'X)pred_matrix.T, where sigma_y is a scalar.
            # Variance of the mean of n_boots predictions is prediction_variance/n_boots
            var_components[n + '_st_err'] = np.sqrt(var_components[n + '_var'] * var_components['x_var'] / n_boots)
            # Normal apprixmation confidence limit
            check_series = (var_components[n + '_in'] - z * var_components[n + "_st_err"] <
                            var_components[n + '_ratio']) & (
                                   var_components[n + "_in"] + z * var_components[n + "_st_err"])
            check_series.name = n + "_ok"
            self.assertTrue(np.all(check_series.values),
                            f"{n} is out of range, on mean of {n_loops}, of {n_boots} bootstraps.")

    def test_get_all_scores(self):
        """
        Test get_all_scores to make sure it returns scores for the values it's supposed to return
        """
        # get_all_scores only works for two categories
        two_categories = pd.concat([self.surrogate_df[["surrogate", "W"]],
                                    1.0 - self.surrogate_df["W"]], axis=1).set_index("surrogate")
        two_categories.columns = ["W", "NW"]
        from jurity.fairness import BinaryFairnessMetrics as bfm
        output_df = bfm.get_all_scores(self.test_data["label"], self.test_data["prediction"], two_categories,
                                       self.test_data["surrogate"], [1])

        from jurity.fairness import BinaryFairnessMetrics

        fairness_funcs = inspect.getmembers(BinaryFairnessMetrics, predicate=inspect.isclass)[:-1]
        for fairness_func in fairness_funcs:
            name = fairness_func[0]
            class_ = getattr(BinaryFairnessMetrics, name)  # grab a class which is a property of BinaryFairnessMetrics
            instance = class_()  # dynamically instantiate such class
            v = output_df.loc[instance.name]["Value"]
            if name in ["AverageOdds", "EqualOpportunity", "FNRDifference", "PredictiveEquality", "StatisticalParity"]:
                self.assertFalse(np.isnan(v), f"Bootstrap returns np.nan for {name}.")
            else:
                self.assertTrue(np.isnan(v), f"Bootstrap not implemented for {name} but returns a value.")
