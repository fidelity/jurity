#Simulations
import pandas as pd
import numpy as np
import math
import sys
sys.path.append('../../jurity/tests')
sys.path.append('../../jurity/jurity')
from jurity.fairness import BinaryFairnessMetrics as bfm
from utils import performance_measures


from test_utils_proba import UtilsProbaSimulator
testing_simulation=False
n_runs=30
avg_counts=[5,10,20,30,40]
num_surrogates=[50,100,300,400,500,1000]
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
surrogates=pd.read_csv('./supporting_data/surrogate_inputs.csv')
surrogates["ZIP"]=surrogates["ZIP"].astype(int)
if testing_simulation:
    output_string = '~/Documents/data/jurity_tests/simulations/sample_size/min_weight_0/{0}_simulation_count_{1}_test_surrogates_{2}.csv'
else:
    output_string = '~/Documents/data/jurity_tests/simulations/sample_size/min_weight_0/{0}_simulation_count_{1}_surrogates_{2}.csv'

def run_one_sim(test_data,membership_df):
    #Sometimes the sub-sampling leads to data errors.
    #Return a dataframe that is all nans in this case.
    #Keep track--if there are too many of these, stop the simulation
    global n_errors
    try:
        oracle_metrics=bfm.get_all_scores(test_data["label"].values,test_data["prediction"].values,
                                    (test_data["class"]=="protected").astype(int).values).rename(columns={"Value":"oracle_value"})
    except:
        oracle_metrics=pd.DataFrame({"oracle_value":[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]},
            index=['Average Odds', 'Disparate Impact', 'Equal Opportunity',
       'FNR difference', 'FOR difference', 'Generalized Entropy Index',
       'Predictive Equality', 'Statistical Parity', 'Theil Index']
        )
        n_errors=n_errors+1
    try:
        prob_metrics=bfm.get_all_scores(test_data["label"],test_data["prediction"],
                        membership_df.set_index("ZIP")[["not_protected","protected"]],
                        test_data["ZIP"],[1]).rename(columns={"Value":"probabilistic_estimate"})
    except:
        prob_metrics=pd.DataFrame({"probabilistic_estimate": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]},
                     index=['Average Odds', 'Disparate Impact', 'Equal Opportunity',
                            'FNR difference', 'FOR difference', 'Generalized Entropy Index',
                            'Predictive Equality', 'Statistical Parity', 'Theil Index']
                     )
        n_errors=n_errors+1
    return pd.concat([oracle_metrics["oracle_value"],prob_metrics["probabilistic_estimate"]], axis=1)

if __name__=="__main__":
    n_errors=0
    rng=np.random.default_rng()
    for sim_label,simulator in scenarios.items():
        for c in avg_counts:
            surrogates["count"]=pd.Series(rng.poisson(lam=c,size=surrogates.shape[0]))
            if testing_simulation:
                test_data=simulator.explode_dataframe(surrogates.head(10))
            else:
                test_data=simulator.explode_dataframe(surrogates)
            print("The number of rows in the data data is: ",test_data.shape)
            for n_surrogates in num_surrogates:
                all_results = []
                for i in range(0, n_runs):
                    #Sample surrogate classes from the dataframe
                    #Take a sample stratified by p(protected) to get a spread
                    #along the x axis for the regression
                    if testing_simulation:
                        sampled_surrogates=surrogates.head(10)["ZIP"].values
                    else:
                        sampled_surrogates=surrogates.groupby("bin").sample(frac=(n_surrogates/surrogates.shape[0]),
                                                                            replace=True)["ZIP"].values
                    #only feed sampled surrogate classes into simulation
                    a=test_data["ZIP"].apply(lambda x: x in sampled_surrogates).values
                    b=surrogates["ZIP"].apply(lambda x:x in sampled_surrogates).values
                    input_data=test_data.iloc[a].copy(deep=True)
                    input_surrogates=surrogates.iloc[b].copy(deep=True)
                    output_df=run_one_sim(input_data,input_surrogates)
                    if n_errors>30:
                        print("Errors limit reached. Stopping simulation.")
                        break
                    output_df["run_id"] = i
                    all_results.append(output_df)
                all_output=pd.concat(all_results)
                all_output["average_count"] = c
                all_output["n_surrogates"] = n_surrogates
                all_output["simulation"] = sim_label
                all_output[~(all_output["probabilistic_estimate"].apply(np.isnan))].to_csv(output_string.format(sim_label, c,n_surrogates))