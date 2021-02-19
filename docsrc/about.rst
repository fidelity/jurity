.. _about:

About Algorithmic Fairness
==========================

Below, see the mathematical definition for each fairness metric in the library.

 - **Average odds** denotes the average of difference in FPR and TPR for group 1 and group 2.

.. math::
    \frac{1}{2} [(FPR_{D = \text{group 1}} - FPR_{D =
    \text{group 2}}) + (TPR_{D = \text{group 2}} - TPR_{D
    = \text{group 1}}))]

- **Disparate Impact** is the ratio of predictions for a "positive" outcome in a binary classification task
        between members of group 1 and group 2, respectively.

.. math::

    \frac{Pr(\hat{Y} = 1 | D = \text{group 1})}
        {Pr(\hat{Y} = 1 | D = \text{group 2})}


Tutorials:

Step-by-step tutorials on algorithmic fairness can be found under the ``examples/notebooks`` folder.

Fairness Metrics Tutorials:

- Quick Start Example: ``examples/notebooks/usage_example_fairness_metrics.ipynb`` presents quick start examples using various metrics.
- Gender as the Protected Attribute: ``examples/notebooks/tutorial_metrics_gender.ipynb`` presents a tutorial on a dataset from the `University of California Irvine Machine Learning Repository <https://archive.ics.uci.edu/ml/index.php>`_ . The learning task is a binary classification of whether an individual has an income over $50k. The ``gender`` attribute serves as the protected attribute.
- Race as the Protected Attribute: ``examples/notebooks/tutorial_race.ipynb`` presents a tutorial on the Propublica Recidivism using a dataset from a non-profit reporting agency `Propublica <https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing>`_ . The learning task is a binary classification of whether an individual will re-offend within the next two years. The ``race`` attribute serves as the protected attribute.

Bias Mitigation Tutorials:

- Quick Start Example: ``examples/notebooks/usage_example_bias_mitigation.ipynb`` presents quick start example of applying bias mitigation to a real-world dataset to achieve fairer predictive outcomes.
- Gender as the protected attribute: ``notebooks/tutorial_mitigation_gender.ipynb`` presents a tutorial on a dataset from the `University of California Irvine Machine Learning Repository <https://archive.ics.uci.edu/ml/index.php>`_ . The learning task is a binary classification of whether an individual has an income over $50k. The ``gender`` attribute serves as the protected attribute.



Make sure you also install the ``notebooks/requirements.txt``
as they depend on other packages that are not needed for the library itself.


About Recommenders Metrics
==========================

For the definitions below, `A` is the set of actual ratings for users and `P` is the set of predictions / recommendations. Each `A_i` and `P_i` represent the list of actual items and list of recommended items respectively for a user `i`.

Some metrics use the relevance function `rel(P_{i,n})`, which is an indicator function that produces `1` if the predicted item at position `n` for user `i` is in the user's relevant set of items.

All the ranking metrics operate on a filtered set of users such that only the users with relevant/clicked items are taken into account. This is in line with industry practices. There is a further filtering for precision related metrics (Precision@k and MAP@k) where each user also has to have a recommendation. This is done to avoid divide by 0 errors.