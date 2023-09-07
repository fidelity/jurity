from typing import NamedTuple
import  numpy as np


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    default_seed = 1
    float_null = np.float64(0.0)
    bootstrap_trials = 100

    TPR = "TPR"
    TNR = "TNR"
    FPR = "FPR"
    FNR = "FNR"
    PPV = "PPV"
    NPV = "NPV"
    FDR = "FDR"
    FOR = "FOR"
    ACC = "ACC"
    PRED_RATE = "Prediction Rate"

    user_id = "user_id"
    item_id = "item_id"
    estimate = "estimate"
    inverse_propensity = "inverse_propensity"
    ips_correction = "ips_correction"
    propensity = "propensity"

    true_positive_ratio = "true_positive_ratio"
    true_negative_ratio = "true_negative_ratio"
    false_positive_ratio = "false_positive_ratio"
    false_negative_ratio = "false_negative_ratio"
    prediction_ratio = "prediction_ratio"
    class_col_name = "class"
    weight_col_name = "count"
    no_label_metrics = ["StatisticalParity", "DisparateImpact"]
    probabilistic_metrics = ["AverageOdds", "EqualOpportunity",
                             "FNRDifference", "StatisticalParity", "PredictiveEquality"]
