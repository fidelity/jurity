# Probabilistic Fairness Demonstration
This folder contains the files necessary to reproduce the results from 
academic papers demonstrating the accuracy of probabilistic 
fairness under different sample sizes. Probabilistic fairness is a technique 
unique to jurity that allows users to calculate fairness metrics when protected status is 
unknown but a surrogate class feature is available. A <i>surrogate</i> class divides 
the population into groups where 
the probability of protected class membership is known given the surrogate class membership.

Probabilistic fairness, its accuracy, and the simulation method used in
these demonstrations are detailed in 
<a href="https://doi.org/10.1007/978-3-031-44505-7_29">"
Surrogate Membership for Inferred Metrics in Fairness Evaluation"</a>

## simulation.py
Demonstrates the accuracy of the probabilistic fairness method, 
showing that the method gives values that are close to the oracle metrics
that would be calculated if protected status were known. 

## simulation_compare_to_model.py
One alternative method for calculating fairness metrics when protected 
status is unknown is to build a predictive model for protected status, assign 
individuals to groups based on model results, and then calculate fairness 
metrics as if protected status were known. This script demonstrates that
fairness metrics calculated in this way are biased, where the degree of the 
bias is based on the PPV (positive predictive value/precision) and NPV
(negative predictive value) of the model for protected status. 

## simulation_counts.py
The performance of probabilistic fairness metrics is related to the number of 
individuals per surrogate class and the number of surrogate classes available. 
This simulation examines performance under different sample size scenarios. 

## Citation 
If you use this analysis in an article, please cite as:
```
@inproceedings{DBLP:conf/lion/ThielbarKZPD23,
  author       = {Melinda Thielbar and
                  Serdar Kadioglu and
                  Chenhui Zhang and
                  Rick Pack and
                  Lukas Dannull},
  editor       = {Meinolf Sellmann and
                  Kevin Tierney},
  title        = {Surrogate Membership for Inferred Metrics in Fairness Evaluation},
  booktitle    = {Learning and Intelligent Optimization - 17th International Conference,
                  {LION} 17, Nice, France, June 4-8, 2023, Revised Selected Papers},
  series       = {Lecture Notes in Computer Science},
  volume       = {14286},
  pages        = {424--442},
  publisher    = {Springer},
  year         = {2023},
  url          = {https://doi.org/10.1007/978-3-031-44505-7_29},
  doi          = {10.1007/978-3-031-44505-7\_29},
  timestamp    = {Thu, 09 Nov 2023 21:13:04 +0100},
  biburl       = {https://dblp.org/rec/conf/lion/ThielbarKZPD23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```