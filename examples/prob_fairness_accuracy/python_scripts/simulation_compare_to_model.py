#Simulationed data: Model-based assignment to protected class vs probabilistic fairness
# One of the claims in the paper is that model-based fairness metrics are biased,
# and that the degree of bias is a function of the PPV (positive predictive value/precision)
# and NPV (negative predctive value) of the models that predicts protected status.
# This simulation demonstrates the difference between probabilistic estimates and
# model-based estimates for a given input data file (located in ../input_data.surrogate_inputscsv)

import pandas as pd
import numpy as np
import math
import sys
sys.path.append('../../tests')
sys.path.append('../../jurity')
from jurity.fairness import BinaryFairnessMetrics as bfm
from constants import Constants
from sklearn.metrics import confusion_matrix
from test_utils_proba import UtilsProbaSimulator

def performance_measures(ground_truth: np.ndarray,
                         predictions: np.ndarray) -> dict:
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

#If true, only simulate a small dataframe. Used to test simulation syntax.
testing_simulation=False
n_runs=30

# The test_utils_proba.py test file in jurity/tests contains a class called
# UtilsProbaSimulator, which can simulate the confusion matrix from an unfair model for different classes.
# Simulation is explained in :

fair_sim=UtilsProbaSimulator({'not_protected': {'pct_positive': 0.2, 'fnr': 0.1, 'fpr': 0.2},'protected': {'pct_positive': 0.2, 'fnr': 0.1, 'fpr': 0.2}},surrogate_name="ZIP")
slightly_unfair_sim=UtilsProbaSimulator({'not_protected': {'pct_positive': 0.2, 'fnr': 0.1, 'fpr': 0.2}, 'protected': {'pct_positive': 0.1, 'fnr': 0.35, 'fpr': 0.1}},surrogate_name="ZIP")
moderately_unfair_sim=UtilsProbaSimulator({'not_protected': {'pct_positive': 0.3, 'fnr': 0.1, 'fpr': 0.3}, 'protected': {'pct_positive': 0.1, 'fnr': 0.45, 'fpr': 0.1}},surrogate_name="ZIP")
very_unfair_sim =UtilsProbaSimulator({'not_protected': {'pct_positive': 0.4, 'fnr': 0.1, 'fpr': 0.3}, 'protected': {'pct_positive': 0.10, 'fnr': 0.65, 'fpr': 0.1}},surrogate_name="ZIP")
extremely_unfair_sim =UtilsProbaSimulator({'not_protected': {'pct_positive': 0.5, 'fnr': 0.1, 'fpr': 0.2}, 'protected': {'pct_positive': 0.10, 'fnr': 0.65, 'fpr': 0.05}},surrogate_name="ZIP")
if testing_simulation:
    scenarios = {"moderately_unfair": moderately_unfair_sim,
           "very_unfair":very_unfair_sim}
else:
    scenarios={"fair":fair_sim,
           "slightly_unfair":slightly_unfair_sim,
           "moderately_unfair":moderately_unfair_sim,
           "very_unfair":very_unfair_sim,
           "extremely_unfair":extremely_unfair_sim}
#Location of input and output files
surrogates=pd.read_csv('../input_data/sampled_surrogate_inputs.csv')
if testing_simulation:
    prob_output_string = '~/Documents/data/jurity_tests/simulations//model_v_prob/{0}_prob_simulation_{1}_surrogates_{2}_count_test.csv'
    model_output_string = '~/Documents/data/jurity_tests/simulations/model_v_prob/{0}_model_simulation_{1}_surrogates_{2}_count_test.csv'
else:
    prob_output_string = '~/Documents/data/jurity_tests/simulations/model_v_prob/{0}_prob_simulation_{1}_surrogates_{2}_count.csv'
    model_output_string = '~/Documents/data/jurity_tests/simulations/model_v_prob/{0}_model_simulation_{1}_surrogates_{2}_count.csv'

def generate_test_data(simulator, membership_df,count_mean,rng=np.random.default_rng()):
    membership_df["count"]=pd.Series(rng.poisson(lam=count_mean,size=membership_df.shape[0]))
    return simulator.explode_dataframe(membership_df)

def calc_prob_estimate(test_data,membership_df):
    oracle_metrics=bfm.get_all_scores(test_data["label"].values,test_data["prediction"].values,
                                    (test_data["class"]=="protected").astype(int).values).rename(columns={"Value":"oracle_value"})
    prob_metrics=bfm.get_all_scores(test_data["label"],test_data["prediction"],
                   membership_df.set_index("ZIP")[["not_protected","protected"]],
                   test_data["ZIP"],[1]).rename(columns={"Value":"probabilistic_estimate"})
    return pd.concat([oracle_metrics["oracle_value"],prob_metrics["probabilistic_estimate"]], axis=1)

def calc_model_estimate(df,rng=np.random.default_rng()):
    out_dfs=[]
    for s in [[0.99, 0.99], [0.9, 0.99], [0.8, 0.9], [0.7, 0.8]]:
        p_given_p = s[0]
        np_given_np = s[1]
        prediction_p=rng.choice([0,1],p=[1-p_given_p,p_given_p],size=df.shape[0])
        prediction_np=rng.choice([0,1],p=[np_given_np,1-np_given_np],size=df.shape[0])
        class_vec_p=(df["class"]=="protected").astype(int).values
        class_vec_np=(df["class"]=="not_protected").astype(int).values
        class_pred=np.multiply(class_vec_p,prediction_p)+np.multiply(class_vec_np,prediction_np)
        scores=bfm.get_all_scores(df["label"].values,df["prediction"].values,class_pred).rename(columns={"Value":"model_estimate"})
        scores["p_given_p"]=p_given_p
        scores["np_given_np"]=np_given_np
        class_model_performance=performance_measures(class_vec_p,class_pred)
        scores["p_given_p"]=p_given_p
        scores["np_given_np"]=np_given_np
        scores["class_PPV"]=class_model_performance[Constants.PPV]
        scores["class_NPV"]=class_model_performance[Constants.NPV]
        scores["class_TPR"]=class_model_performance[Constants.TPR]
        scores["class_BR"]=np.sum(class_vec_p)
        out_dfs.append(scores.reset_index()[["Metric","model_estimate","class_PPV","class_NPV","class_TPR","p_given_p","np_given_np"]])
    return pd.concat(out_dfs,axis=0)

if __name__=="__main__":
    n_surrogates=surrogates.shape[0]
    generator=np.random.default_rng()
    for sim_label,simulator in scenarios.items():
        prob_results=[]
        model_results=[]
        for i in range(0, n_runs):
            if testing_simulation:
                test_df = generate_test_data(simulator, surrogates, 50, generator)
            else:
                test_df = generate_test_data(simulator, surrogates, 50, generator)
            p=calc_prob_estimate(test_df,surrogates)
            p["run_id"]=i
            prob_results.append(p)
            m=calc_model_estimate(test_df, generator)
            m["run_id"]=i
            model_results.append(m)
        all_prob_results=pd.concat(prob_results,axis=0)
        all_prob_results["simulation"]=sim_label
        all_model_results=pd.concat(model_results,axis=0)
        all_model_results["simulation"]=sim_label
        all_prob_results.to_csv(prob_output_string.format(sim_label,50,surrogates.shape[0]))
        all_model_results.to_csv(model_output_string.format(sim_label,50,surrogates.shape[0]),index=False)