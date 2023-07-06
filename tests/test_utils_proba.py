import unittest
import sklearn
import pandas as pd
import numpy as np
from scipy.stats import kstest
from jurity.utils_proba import BiasCalculator, BiasCalcFromDataFrame, SummaryData


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
        cls.test_data = pd.DataFrame({"Surrogate": [1, 2, 3, 4, 5],
                                       "W": [0.5, 0.1, 0.3, 0.7, 0.6],
                                       "B": [0.2, 0.7, 0.5, 0.1, 0.3],
                                       "O": [0.3, 0.2, 0.2, 0.2, 0.1],
                                       "N": [30, 10, 6, 9, 2],
                                       "false_positive_ratio": [0.1, 0.5, 0.3, 0.1, 0.2],
                                       "true_positive_ratio": [0.9, 0.5, 0.7, 0.9, 0.8],
                                       "false_negative_ratio": [0.2, 0.4, 0.3, 0.2, 0.1],
                                       "true_negative_ratio": [0.8, 0.6, 0.7, 0.8, 0.9]})

        cls.bcfd = BiasCalcFromDataFrame(["W", "B", "O"], "N", [1,2],
                                          ["false_positive_ratio", "true_positive_ratio", "false_negative_ratio",
                                          "true_negative_ratio"])

        cls.bc = cls.bcfd.get_bias_calculator(cls.test_data, 0)

        # TODO this seems not used? then remove? OR comment out with a note for values below as reference
        test_boot_results = pd.DataFrame({"false_negative_ratio": [0.25, 0.7, 0.3],
                                          "false_positive_ratio": [0.3, 0.4, 0.5],
                                          "true_negative_ratio": [0.75, 0.3, 0.7],
                                          "true_positive_ratio": [0.7, 0.6, 0.5]},
                                         index=["W", "B", "O"])

        # Test Data for Bootstrap simulations
        # TODO shall we call it surrogate_df instead of zip df?
        cls.zip_df = pd.DataFrame({"ZIP5_AD_IMP": list(range(0, 99)),
                                    "count": [34, 32, 26, 34, 37, 23, 27, 21, 31, 28, 38, 18, 25, 30, 26, 35, 31,
                                              23, 28, 28, 22, 33, 30, 19, 35, 25, 24, 31, 25, 42, 27, 23, 32, 36,
                                              25, 37, 24, 15, 39, 28, 26, 38, 36, 30, 27, 27, 33, 28, 23, 24,
                                              30, 23, 35, 29, 35, 34, 27, 20, 27, 21, 25, 27, 26, 32, 30, 30, 32,
                                              35, 24, 31, 34, 37, 26, 20, 30, 18, 31, 27, 23, 41, 28, 30, 33, 34,
                                              35, 30, 24, 24, 25, 30, 26, 31, 26, 25, 25, 35, 30, 25, 20],
                                    "pct_w_s": [0.36137754, 0.83653862, 0.98303716, 0.52943704, 0.80254777,
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
                                    "pct_b_s": [0.53346927, 0.10651034, 0.00565428, 0.31113021, 0.00636943,
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
                                    "pct_a_s": [5.91596800e-03, 2.47713860e-02, 8.07754000e-04, 2.57842700e-03,
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
                                    "pct_ai_s": [0.00289505, 0.00327419, 0.00080775, 0.00673256, 0.01910828,
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
                                    "pct_o_s": [0.08065856, 0.00486167, 0.00323102, 0.12304827, 0.02547771,
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
                                    "pct_2_s": [0.01568361, 0.02404379, 0.00646204, 0.02707349, 0.08917198,
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

        cls.zip_df["pct_nw_s"] = 1 - cls.zip_df["pct_w_s"]

    def test_calc_one_bag_form(self):
        """
        The calling functions for calc_one_bag expects the output in a certain format.
        Make sure the return is as expected.
        --There is the right number of rows and columns
        --Multiple Y's are handled appropriately
        --Single Y's are handed appropriately
        """
        out = self.bc.calc_one_bag(self.test_data[["W", "B", "O"]].to_numpy(),
                                   self.test_data[["false_positive_ratio", "false_negative_ratio",
                                                   "true_positive_ratio", "true_negative_ratio"]].to_numpy(),
                                   self.test_data["N"].to_numpy())

        self.assertTrue(isinstance(out["false_positive_ratio"], sklearn.linear_model.LinearRegression))
        self.assertTrue(isinstance(out["false_negative_ratio"], sklearn.linear_model.LinearRegression))
        self.assertTrue(isinstance(out["true_positive_ratio"], sklearn.linear_model.LinearRegression))
        self.assertTrue(isinstance(out["true_negative_ratio"], sklearn.linear_model.LinearRegression))

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
        out = self.bc.calc_one_bag(self.test_data[["B", "O"]].to_numpy(),
                                   self.test_data[["false_positive_ratio", "true_positive_ratio",
                                                   "false_negative_ratio", "true_negative_ratio"]].to_numpy(),
                                   self.test_data["N"].to_numpy())

        fp_model = sklearn.linear_model.LinearRegression()
        fp_model.fit(self.test_data[["B", "O"]].to_numpy(),
                     self.test_data["false_positive_ratio"].to_numpy(),
                     self.test_data["N"].to_numpy())

        np.testing.assert_array_almost_equal(fp_model.coef_, out["false_positive_ratio"].coef_)

        fn_model = sklearn.linear_model.LinearRegression()
        fn_model.fit(self.test_data[["B", "O"]].to_numpy(),
                     self.test_data["false_negative_ratio"].to_numpy(),
                     self.test_data["N"].to_numpy())

        np.testing.assert_array_almost_equal(fn_model.coef_, out["false_negative_ratio"].coef_)

        tp_model = sklearn.linear_model.LinearRegression()
        tp_model.fit(self.test_data[["B", "O"]].to_numpy(),
                     self.test_data["true_positive_ratio"].to_numpy(),
                     self.test_data["N"].to_numpy())

        np.testing.assert_array_almost_equal(tp_model.coef_, out["true_positive_ratio"].coef_)

        tn_model = sklearn.linear_model.LinearRegression()
        tn_model.fit(self.test_data[["B", "O"]].to_numpy(),
                     self.test_data["true_negative_ratio"].to_numpy(),
                     self.test_data["N"].to_numpy())

        np.testing.assert_array_almost_equal(tn_model.coef_, out["true_negative_ratio"].coef_)

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
        self.assertEqual(bootstrap.shape, (20, 5))

    def test_transform_bootstrap_results_form(self):
        """
        Test that transform_results gives an output with the correct number of rows and columns
        """
        bootstrap = self.bc.run_bootstrap(5)
        self.assertEqual(self.bc.transform_bootstrap_results(bootstrap).shape, (3, 9))

    def test_transform_bootstrap_results_answer(self):
        """
        Test that transform_bootstrap_results gives the correct numbers
        """
        boot = self.bc.run_bootstrap(5)
        ratios_added = boot.groupby("stat_name").mean()
        del ratios_added["run_id"]
        ratios_added = ratios_added.T
        for i in range(1, ratios_added.shape[0]):
            ratios_added.iloc[i] += ratios_added.iloc[0]
        ratios_added = ratios_added.to_numpy()
        ans = np.array([ratios_added[:, 1] / (ratios_added[:, 1] + ratios_added[:, 2]),
                        ratios_added[:, 0] / (ratios_added[:, 0] + ratios_added[:, 3]),
                        ratios_added[:, 3] / (ratios_added[:, 3] + ratios_added[:, 0]),
                        ratios_added[:, 2] / (ratios_added[:, 2] + ratios_added[:, 1])])
        trans = self.bc.transform_bootstrap_results(boot)[["FPR", "FNR", "TPR", "TNR"]]
        np.testing.assert_almost_equal(trans, ans.T)

    """
    These tests are for MakeBiasCalcFromDataFrame 
    """

    def test_make_Y_matrix(self):
        """
        Test that make_x_matrix creates a two-dimensional numpy array with the correct numbers
        """
        Y = self.bcfd.get_Y_matrix(self.test_data)
        ans = [[0.1, 0.9, 0.2, 0.8],
               [0.5, 0.5, 0.4, 0.6],
               [0.3, 0.7, 0.3, 0.7],
               [0.1, 0.9, 0.2, 0.8],
               [0.2, 0.8, 0.1, 0.9]]
        np.testing.assert_array_almost_equal(Y, ans)

    def test_make_X_matrix(self):
        """
        test that make_X_matrix gives a two-dimensional array with the correct numbers
        """
        X = self.bcfd.get_X_matrix(self.test_data)
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
        w = self.bcfd.get_W_array(self.test_data)
        ans = [30, 10, 6, 9, 2]
        np.testing.assert_array_equal(w, ans)

    def test_make_bias_calculator_filter(self):
        """
        Test that make_bias_calculator filters rows with small counts
        """
        bc_filtered = self.bcfd.get_bias_calculator(self.test_data, 7)
        self.assertEqual(bc_filtered.X().shape[0], 3)

    def test_make_bias_calculator_names(self):
        """
        Test to make certain make_bias_calculator has the correct race labels and omitted category
        """
        self.assertTrue("B" in self.bc.surrogate_labels()[1])
        self.assertTrue("O" in self.bc.surrogate_labels()[1])
        self.assertFalse("W" in self.bc.surrogate_labels()[1])
        self.assertEqual("W", self.bc.surrogate_labels()[0][0])

    def test_bias_maker_bad_data(self):
        # duplicates
        self.assertRaises(ValueError, BiasCalcFromDataFrame, ["W", "W", "B", "O"], "N", [3,4],
                          ["false_positive_ratio", "true_positive_ratio", "false_negative_ratio",
                           "true_negative_ratio"])
        self.assertRaises(ValueError, BiasCalcFromDataFrame, ["W", "B", "O"], "N", [1,2],
                          ["false_positive_ratio", "false_positive_ratio", "true_positive_ratio",
                           "false_negative_ratio", "true_negative_ratio"])
        # not list
        self.assertRaises(ValueError, BiasCalcFromDataFrame, "B", "N", [1],
                          ["false_positive_ratio", "true_positive_ratio", "false_negative_ratio",
                           "true_negative_ratio"])
        self.assertRaises(ValueError, BiasCalcFromDataFrame, ["W", "B", "O"], "N", [1,2],
                          "false_positive_ratio")
        # not string
        self.assertRaises(ValueError, BiasCalcFromDataFrame, ["W", "B", "O", 0], "N", [1,2],
                          ["false_positive_ratio", "true_positive_ratio", "false_negative_ratio",
                           "true_negative_ratio"])
        self.assertRaises(ValueError, BiasCalcFromDataFrame, ["W", "B", "O"], "N", [1,2],
                          ["false_positive_ratio", "true_positive_ratio", "false_negative_ratio", "true_negative_ratio",
                           0])
        # column missing
        fac = BiasCalcFromDataFrame(["W", "B", "O", "hello world"], "N", [1,2],
                                    ["false_positive_ratio", "true_positive_ratio", "false_negative_ratio",
                                     "true_negative_ratio"])
        self.assertRaises(ValueError, fac.get_bias_calculator, self.test_data, 1)
        fac = BiasCalcFromDataFrame(["W", "B", "O"], "N", [1,2],
                                    ["false_positive_ratio", "true_positive_ratio", "false_negative_ratio",
                                     "true_negative_ratio", "hello world"])
        self.assertRaises(ValueError, fac.get_bias_calculator, self.test_data, 1)
        fac = BiasCalcFromDataFrame(["W", "B", "O"], "hello world", [1,2],
                                    ["false_positive_ratio", "true_positive_ratio", "false_negative_ratio",
                                     "true_negative_ratio"])
        self.assertRaises(ValueError, fac.get_bias_calculator, self.test_data, 1)

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

        with_summary = summarize.confusion_matrix_actual(test_df, "prediction", "labels")

        result = pd.concat([pd.Series(surrogates, name="surrogates"), with_summary], axis=1).groupby("surrogates").mean()

        result.columns = ["_".join([c, "ratio"]) for c in result.columns]

        df = SummaryData.summarize(predictions, memberships, surrogates, labels)

        self.assertEqual(df.shape[0], len(set(surrogates)),
                         "MakeSummaryData.summarize_detail returns wrong number of rows.")

        self.assertTrue(df[["true_positive_ratio", "true_negative_ratio", "false_positive_ratio",
                            "false_negative_ratio", "prediction_ratio"]].equals(result.drop("correct_ratio", axis=1)))

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

        bc = BiasCalculator.from_df(df,
                                    [1],
                                    ["A","B"],
                                    test_names=["true_positive_ratio",
                                                "true_negative_ratio",
                                                "false_positive_ratio",
                                                "false_negative_ratio"])

        self.assertTrue(np.all(np.isclose(bc.X().ravel(), df["B"].values)),
                        "X matrix in bias calculator does not match race probabilities."
                        "\nBias calc has:{0},original df has: {1}.\nCompare result is: {2}".format(bc.X().ravel(), df[
                            "B"].values, bc.X().ravel() == (df["B"].values)))

        self.assertEqual(bc.Y().shape[0], len(np.unique(surrogates)),
                         "Y matrix in BiasCalculator has wrong length. Length is {0}. Should be {1}.".format(
                             bc.Y().shape[0], len(np.unique(surrogates))))

    """
    Helper functions 
    """

    @staticmethod
    def explode_dataframe(df, count_name="count"):
        """
        Given a dataframe that has a count, produce a number of identical rows equal to that count
        """
        df2 = df.loc[df.index.repeat(df[count_name])]
        return df2.drop("count", axis=1)

    # For the simulation, build "True" race groups based on population
    # Note; The census data names the columns as follows:
    # pct_white_zip, pct_black_zip, etc
    # TODO: Need to use the census labels as keys in a dictionary or use some other method
    # So we can continue to use this when the census data changes.
    # TODO: REMOVE race terminology
    @staticmethod
    def assign_race(population_data, generator, zip_col_name='ZIP5_AD_IMP'):
        # Passing in the global random number generator
        # Lets us make sure that we're not accidentally resetting the seed
        zip_race_prob_grouped = population_data.groupby(zip_col_name)
        zip_groups = []
        for name, group in zip_race_prob_grouped:
            try:
                probs = [group["pct_w_s"].unique()[0], group.pct_black_zip.unique()[0],
                         group.pct_api_zip.unique()[0],
                         group.pct_aian_zip.unique()[0], group.pct_other_zip.unique()[0],
                         group.pct_2prace_zip.unique()[0]]
                group['surrogate'] = generator.choice(
                    ["W", "B", 'A',
                     'AI',
                     'O', 'Two'], len(group), p=probs)
            except:
                pass
                # print(name)
                # print(probs)
            zip_groups.append(group)
        out_data = pd.concat(zip_groups)
        return out_data

    @staticmethod
    # TODO REMOVE race terminology
    def assign_race_and_accuracy(input_data, rates_by_race, generator, race_col_name="w"):
        # Assign everyone a "true" race for simulation purposes
        race_assignments = TestUtilsProba.assign_race(input_data, generator)
        # Current simulation only handles 2 categories: white or not.
        race_assignments["w"] = np.where(race_assignments["surrogate"] == "W", "W", "NW")
        # Assign each individual a quadrant in the confusion matrix based on:
        #   Percent of positive (not predict_pos_probs)
        #   probability of being a false positive
        #   probability of being a false negative
        # These are different by race and fed into the simulation through indexes
        # Index keys are the values in the race column, e.g. "White" and "Non-White"
        model_outcomes = TestUtilsProba.model_outcome_by_race(race_assignments, rates_by_race,
                                                              race_col_name=race_col_name)
        accuracy = TestUtilsProba.accuracy_columns(model_outcomes, "prediction", "label")
        return accuracy

    @staticmethod
    # TODO I feel we are duplicating A LOT confustion matrix, rate calculation functions, no?
    def confusion_matrix_prob(percent_positive, fpr, fnr, verbose=False):
        """
        # This is the probability that the person is labeled as positive in the data
        Calculate the % of False Positive, False Negative, True Negative, and True Positive in total based on predefined inputs.
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

    # TODO: sk: isn't this function existing/repeated somewhere else, see utils_proba.calc_rates() function.?
    # See also what's avail in utils.
    @staticmethod
    def calc_rates(results):
        results["FPR"] = results["false_positive_ratio"] / (
                results["false_positive_ratio"] + results["true_negative_ratio"])
        results["FNR"] = results["false_negative_ratio"] / (
                results["false_negative_ratio"] + results["true_positive_ratio"])
        results["TPR"] = results["true_positive_ratio"] / (
                results["true_positive_ratio"] + results["false_negative_ratio"])
        results["TNR"] = results["true_negative_ratio"] / (
                results["true_negative_ratio"] + results["false_positive_ratio"])
        return results

    # TODO REMOVE race terminology
    @staticmethod
    def model_outcome_by_race(zip_race_assignment, rates_by_race, race_col_name="w"):
        # Assign true positive, true negative, etc by race
        zip_race_prob_grouped = zip_race_assignment.groupby(race_col_name)

        classified_groups = []
        for name, group in zip_race_prob_grouped:
            rates_dict = rates_by_race[name]
            probs = TestUtilsProba.confusion_matrix_prob(rates_dict["pct_positive"], rates_dict["fpr"],
                                                         rates_dict["fnr"])
            group['pred_category'] = np.random.choice(['fp', 'fn', 'tn', 'tp'], len(group), p=probs)
            group['label'] = np.where(group['pred_category'].isin(['tp', 'fn']), 1, 0)
            group['prediction'] = np.where(group['pred_category'].isin(['tp', 'fp']), 1, 0)
            classified_groups.append(group)
        classified_data = pd.concat(classified_groups)
        return classified_data

    @staticmethod
    # Add columns to a pandas dataframe flagging each row as false positive, etc.
    def accuracy_columns(test_data, pred_col, label_col):
        test_data["correct"] = (test_data[pred_col] == test_data[label_col]).astype(int)
        test_data["true_positive"] = (test_data["correct"] & (test_data[label_col] == 1)).astype(int)
        test_data["true_negative"] = (test_data["correct"] & (test_data[label_col] == 0)).astype(int)
        test_data["false_negative"] = ((test_data["correct"] == False) & (test_data[pred_col] == 0)).astype(int)
        test_data["false_positive"] = ((test_data["correct"] == False) & (test_data[pred_col] == 1))
        return test_data
