#Simulations
import pandas as pd
import numpy as np
import math
import sys
sys.path.append('../../jurity/tests')
sys.path.append('../../jurity/jurity')
from jurity.fairness import BinaryFairnessMetrics as bfm
from test_utils_proba import UtilsProbaSimulator

output_path='~/Documents/data/jurity_tests/simulations/'

testing_simulation=False
n_runs=30
avg_counts=[30,50]
fair_sim=UtilsProbaSimulator({'not_protected': {'pct_positive': 0.2, 'fnr': 0.1, 'fpr': 0.2},'protected': {'pct_positive': 0.2, 'fnr': 0.1, 'fpr': 0.2}},surrogate_name="ZIP")
slightly_unfair_sim=UtilsProbaSimulator({'not_protected': {'pct_positive': 0.2, 'fnr': 0.1, 'fpr': 0.2}, 'protected': {'pct_positive': 0.1, 'fnr': 0.35, 'fpr': 0.1}},surrogate_name="ZIP")
moderately_unfair_sim=UtilsProbaSimulator({'not_protected': {'pct_positive': 0.3, 'fnr': 0.1, 'fpr': 0.3}, 'protected': {'pct_positive': 0.1, 'fnr': 0.45, 'fpr': 0.1}},surrogate_name="ZIP")
very_unfair_sim =UtilsProbaSimulator({'not_protected': {'pct_positive': 0.4, 'fnr': 0.1, 'fpr': 0.3}, 'protected': {'pct_positive': 0.10, 'fnr': 0.65, 'fpr': 0.1}},surrogate_name="ZIP")
extremely_unfair_sim =UtilsProbaSimulator({'not_protected': {'pct_positive': 0.5, 'fnr': 0.1, 'fpr': 0.2}, 'protected': {'pct_positive': 0.10, 'fnr': 0.65, 'fpr': 0.05}},surrogate_name="ZIP")

scenarios={"fair":fair_sim,
           "slightly_unfair":slightly_unfair_sim,
           "moderately_unfair":moderately_unfair_sim,
           "very_unfair":very_unfair_sim,
           "extremely_unfair":extremely_unfair_sim}
surrogates=pd.read_csv('../input_data/surrogate_inputs.csv')
if testing_simulation:
    output_string = output_path+'{0}_simulation_count_{1}_surrogates_{2}_test.csv'
else:
    output_string = output_path+'{0}_simulation_count_{1}_surrogates_{2}.csv'

def run_one_sim(simulator, membership_df,count_mean,rng=np.random.default_rng()):
    membership_df["count"]=pd.Series(rng.poisson(lam=count_mean,size=membership_df.shape[0]))
    test_data=simulator.explode_dataframe(membership_df)
    oracle_metrics=bfm.get_all_scores(test_data["label"].values,test_data["prediction"].values,
                                    (test_data["class"]=="protected").astype(int).values).rename(columns={"Value":"oracle_value"})
    prob_metrics=bfm.get_all_scores(test_data["label"],test_data["prediction"],
                   membership_df.set_index("ZIP")[["not_protected","protected"]],
                   test_data["ZIP"],[1]).rename(columns={"Value":"probabilistic_estimate"})
    predicted_class=test_data[["not_protected","protected"]].values.tolist()
    argmax_metrics=bfm.get_all_scores(test_data["label"].values,test_data["prediction"].values,
                                     predicted_class).rename(columns={"Value":"argmax_estimate"})
    return pd.concat([oracle_metrics["oracle_value"],prob_metrics["probabilistic_estimate"], argmax_metrics["argmax_estimate"]], axis=1)

if __name__=="__main__":
    n_surrogates=surrogates.shape[0]
    for sim_label,simulator in scenarios.items():
        for c in avg_counts:
            all_results=[]
            for i in range(0, n_runs):
                if testing_simulation:
                    output_df = run_one_sim(simulator, surrogates.head(10), c)
                else:
                    output_df = run_one_sim(simulator, surrogates, c)
                output_df["run_id"] = i
                all_results.append(output_df)
            all_output=pd.concat(all_results)
            all_output["average_count"] = c
            all_output["simulation"] = sim_label
            all_output["n_surrogates"] = n_surrogates
            all_output[~(all_output["probabilistic_estimate"].apply(np.isnan))].to_csv(output_string.format(sim_label, c,n_surrogates))