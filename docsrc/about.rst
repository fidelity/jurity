.. _about:

About Algorithmic Fairness
==========================

Below, see the mathematical definition for each of the fairness metrics in the library.

- Average odds denotes the average of difference in FPR and TPR for group 1 and group 2.

.. math::
    \frac{1}{2} [(FPR_{D = \text{group 1}} - FPR_{D =
    \text{group 2}}) + (TPR_{D = \text{group 2}} - TPR_{D
    = \text{group 1}}))]

- Disparate Impact is the ratio of predictions for a "positive" outcome in a binary classification task between members of group 1 and group 2, respectively.

.. math::

    \frac{P(\hat{Y} = 1 | D = \text{group 1})}
        {P(\hat{Y} = 1 | D = \text{group 2})}

- Equal Opportunity calculates the ratio of true positives to positive examples in the dataset, :math:`TPR = TP/P`, conditioned on a protected attribute.

- FNR Difference measures the equality (or lack thereof) of the false negative rates across groups. In practice, this metric is implemented as a difference between the metric value for group 1 and group 2.

.. math::

    E[d(X)=0 \mid Y=1, g(X)] = E[d(X)=0, Y=1]

- Generalized entropy index is proposed as a unified individual and group fairness measure in [3]_. With :math:`b_i = \hat{y}_i - y_i + 1`:

.. math::

           \mathcal{E}(\alpha) = \begin{cases}
              \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right] &
              \alpha \ne 0, 1, \\
              \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu} & \alpha=1, \\
            -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
            \end{cases}

References:
            .. [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
             A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via
             Inequality Indices, ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.

- Predictive Equality is defined as the situation when accuracy of decisions is equal across two groups, as measured by false positive rate (FPR).

.. math::

    E[d(X)|Y=0, g(X)] = E[d(X), Y=0]
    
- Statistical Parity measures the difference in probabilities of a positive outcome across two groups.
 
.. math::

    P(\hat{Y} = 1 | D = \text{group 1}) - P(\hat{Y} = 1 | D = \text{group 2})

- Theil Index is the generalized entropy index with :math:`\alpha = 1`. See Generalized Entropy index.


About Recommenders Metrics
==========================

Jurity offers various standardized metrics for measuring the recommendation performance.
While the recommendation systems community agrees on these metrics, the implementations can be different, especially when it comes to edge-cases.

For the definitions below, :math:`A` is the set of actual ratings for users and :math:`P` is the set of predictions / recommendations.
Each :math:`A_i` and :math:`P_i` represents the list of actual items and list of recommended items, respectively, for a user :math:`i`.


Binary Recommender Metrics
--------------------------
Binary recommender metrics directly measure the click interaction.

- **CTR**: Let :math:`M` denote the set of user-item pairs that appear both in actual ratings and recommendations, and :math:`C(M_i)` be an indicator function that procudes :math:`1` if the user clicked on the item, and :math:`0` if they didn't.

.. math::
    CTR = \frac{1}{\left | M \right |}\sum_{i=1}^{\left | M \right |} C(M_i)

Ranking Recommender Metrics
---------------------------
Ranking metrics reward putting the clicked items on a higher position/rank than the others.
Some metrics use the relevance function :math:`rel(P_{i,n})`, which is an indicator function that produces :math:`1` if the predicted item at position :math:`n` for user :math:`i` is in the user's relevant set of items.

All the ranking metrics operate on a filtered set of users such that only the users with relevant/clicked items are taken into account.
This is in line with industry practices.
There is a further filtering for precision related metrics (Precision@k and MAP@k) where each user also has to have a recommendation.
This is done to avoid divide by 0 errors.

- **Precision**

.. math::
    Precision@k = \frac{1}{\left | A \cap P \right |}\sum_{i=1}^{\left | A \cap P \right |} \frac{\left | A_i \cap P_i[1:k] \right |}{\left | P_i[1:k] \right |}

- **Recall**

.. math::
    Recall@k = \frac{1}{\left | A \right |}\sum_{i=1}^{\left | A \right |} \frac{\left | A_i \cap P_i[1:k] \right |}{\left | A_i \right |}

- **MAP: Mean Average Precision**

.. math::
    MAP@k = \frac{1}{\left | A \right |} \sum_{i=1}^{\left | A \right |} \frac{1}{min(k,\left | A_i \right |))}\sum_{n=1}^k Precision_i(n) \times rel(P_{i,n})

- **NDCG: Normalized Discounted Cumulative Gain**

.. math::
    NDCG@k = \frac{1}{\left | A \right |} \sum_{i=1}^{\left | A \right |} \frac {\sum_{r=1}^{\left | P_i \right |} \frac{rel(P_{i,r})}{log_2(r+1)}}{\sum_{r=1}^{\left | A_i \right |} \frac{1}{log_2(r+1)}}
