.. _about_fairness:

About Algorithmic Fairness
==========================

Below, see the mathematical definition for each of the fairness metrics in the library.

Average Odds
^^^^^^^^^^^^
Average Odds denotes the average of difference in FPR and TPR for group 1 and group 2.

.. math::
    \frac{1}{2} [(FPR_{D = \text{group 1}} - FPR_{D =
    \text{group 2}}) + (TPR_{D = \text{group 2}} - TPR_{D
    = \text{group 1}}))]

Disparate Impact
^^^^^^^^^^^^^^^^
Disparate Impact is the ratio of predictions for a "positive" outcome in a binary classification task between members of group 1 and group 2, respectively.

.. math::

    \frac{P(\hat{Y} = 1 | D = \text{group 1})}
        {P(\hat{Y} = 1 | D = \text{group 2})}

Equal Opportunity
^^^^^^^^^^^^^^^^^
Equal Opportunity calculates the ratio of true positives to positive examples in the dataset, :math:`TPR = TP/P`, conditioned on a protected attribute.

FNR Difference
^^^^^^^^^^^^^^
FNR Difference measures the equality (or lack thereof) of the false negative rates across groups. In practice, this metric is implemented as a difference between the metric value for group 1 and group 2.

.. math::

    E[d(X)=0 \mid Y=1, g(X)] = E[d(X)=0, Y=1]

Generalized Entropy Index
^^^^^^^^^^^^^^^^^^^^^^^^^
Generalized Entropy Index is proposed as a unified individual and group fairness measure in [1]_. With :math:`b_i = \hat{y}_i - y_i + 1`:

.. math::

           \mathcal{E}(\alpha) = \begin{cases}
              \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right] &
              \alpha \ne 0, 1, \\
              \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu} & \alpha=1, \\
            -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
            \end{cases}

References:
            .. [1] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
             A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via
             Inequality Indices, ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.

Predictive Equality
^^^^^^^^^^^^^^^^^^^
Predictive Equality is defined as the situation when accuracy of decisions is equal across two groups, as measured by false positive rate (FPR).

.. math::

    E[d(X)|Y=0, g(X)] = E[d(X), Y=0]

Statistical Parity
^^^^^^^^^^^^^^^^^^
Statistical Parity measures the difference in probabilities of a positive outcome across two groups.
 
.. math::

    P(\hat{Y} = 1 | D = \text{group 1}) - P(\hat{Y} = 1 | D = \text{group 2})

Theil Index
^^^^^^^^^^^
Theil Index is the generalized entropy index with :math:`\alpha = 1`. See Generalized Entropy Index.


Equalized Odds
^^^^^^^^^^^^^^

Equalized odds is a bias mitigation technique where subset of decisions of a binary classifier is flipped at uniform random in each of two groups to achieve equality of TPR and FPR across the two groups as proposed in [2]_. This subset rate in each group is learned via constrained optimization.

References:
            .. [2] Moritz Hardt, Eric Price, and Nathan Srebro. 2016. Equality of opportunity in supervised learning. In Proceedings of the 30th International Conference on Neural Information Processing Systems (NIPS'16).
