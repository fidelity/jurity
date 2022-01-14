.. _about_reco:

About Recommenders Metrics
==========================

Jurity offers various standardized metrics for measuring the recommendation performance.
While the recommendation systems community agrees on these metrics, the implementations can be different, especially when it comes to edge-cases.

For the definitions below, :math:`A` is the set of actual ratings for users and :math:`P` is the set of predictions / recommendations.
Each :math:`A_i` and :math:`P_i` represents the list of actual items and list of recommended items, respectively, for a user :math:`i`.


Binary Recommender Metrics
--------------------------
Binary recommender metrics directly measure the click interaction.

CTR: Click-through Rate
^^^^^^^^^^^^^^^^^^^^^^^

CTR offers three reward estimation methods.

Direct estimation ("matching") measures the accuracy of the recommendations over the subset of user-item pairs that appear in both actual ratings and recommendations.

Let :math:`M` denote the set of user-item pairs that appear in both actual ratings and recommendations, and :math:`C(M_i)` be an indicator function that produces :math:`1` if the user clicked on the item, and :math:`0` if they didn't.

.. math::
    CTR = \frac{1}{\left | M \right |}\sum_{i=1}^{\left | M \right |} C(M_i)

Inverse propensity scoring (IPS) weights the items by how likely they were to be recommended by the historic policy
if the user saw the item in the historic data. Due to the probability inversion, less likely items are given more weight.

.. math::
    IPS = \frac{1}{n} \sum r_a \times \frac{I(\hat{a} = a)}{P(a|x,h)}

In this calculation: n is the total size of the test data; :math:`r_a` is the observed reward;
:math:`\hat{a}` is the recommended item; :math:`I(\hat{a} = a)` is a boolean of whether the user-item pair has
historic data; and :math:`P(a|x,h)` is the probability of the item being recommended for the test context given
the historic data.

Doubly robust estimation (DR) combines the directly predicted values with a correction based on how
likely an item was to be recommended by the historic policy if the user saw the item in the historic data.

.. math::
    DR = \frac{1}{n} \sum \hat{r}_a + \frac{(r_a -\hat{r}_a) I(\hat{a} = a)}{p(a|x,h)}

In this calculation, :math:`\hat{r}_a` is the predicted reward.

At a high level, doubly robust estimation combines a direct estimate with an IPS-like correction if historic data is
available. If historic data is not available, the second term is 0 and only the predicted reward is used for the
user-item pair.


The IPS and DR implementations are based on: Dudík, Miroslav, John Langford, and Lihong Li.
"Doubly robust policy evaluation and learning." Proceedings of the 28th International Conference on International
Conference on Machine Learning. 2011. Available as arXiv preprint arXiv:1103.4601 

AUC: Area Under the Curve
^^^^^^^^^^^^^^^^^^^^^^^^^

AUC is the probability that the recommender will rank a randomly chosen relevant/clicked instance higher than a randomly chosen non-relevant/not-clicked one over the subset of user-item pairs that appear in both actual ratings and recommendations.

Let :math:`M` denote the set of user-item pairs that appear in both actual ratings and recommendations with :math:`M^1` the set of relevant/clicked instances and :math:`M^0` the non-clicked instances.
If :math:`f(t)` is the score returned by the recommender for the :math:`t`-th instance then:

.. math::
    AUC = \frac{\sum_{t_0 \in M^0}\sum_{t_1 \in M^1}I[(f(t_0) < f(t_1)]}{\left | M^0 \right | \left | M^1 \right |}

Ranking Recommender Metrics
---------------------------
Ranking metrics reward putting the clicked items on a higher position/rank than the others.
Some metrics use the relevance function :math:`rel(P_{i,n})`, which is an indicator function that produces :math:`1` if the predicted item at position :math:`n` for user :math:`i` is in the user's relevant set of items.

All the ranking metrics operate on a filtered set of users such that only the users with relevant/clicked items are taken into account.
This is in line with industry practices.
There is a further filtering for precision related metrics (Precision@k and MAP@k) where each user also has to have a recommendation.
This is done to avoid divide by 0 errors.

Precision
^^^^^^^^^
Precision@k measures how consistently a model is able to pinpoint the items a user would interact with.
A recommender system that only provides recommendations for 5% of the users will have a high precision if the users receiving the recommendations always interact with them.

.. math::
    Precision@k = \frac{1}{\left | A \cap P \right |}\sum_{i=1}^{\left | A \cap P \right |} \frac{\left | A_i \cap P_i[1:k] \right |}{\left | P_i[1:k] \right |}

Recall
^^^^^^
Recall@k measures whether a model can capture all the items the user has interacted with.
If __k__ is high enough, a recommender system can get a high recall even if it has a large amount of irrelevant recommendations, if it has also identified the relevant recommendations.

.. math::
    Recall@k = \frac{1}{\left | A \right |}\sum_{i=1}^{\left | A \right |} \frac{\left | A_i \cap P_i[1:k] \right |}{\left | A_i \right |}

MAP: Mean Average Precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^
MAP@k measures a position-sensitive version of Precision@k, where getting the top-most recommendations more precise has a more important effect than getting the last recommendations correct.
When :math:`k=1`, Precision@k and MAP@k are the same.

.. math::
    MAP@k = \frac{1}{\left | A \right |} \sum_{i=1}^{\left | A \right |} \frac{1}{min(k,\left | A_i \right |))}\sum_{n=1}^k Precision_i(n) \times rel(P_{i,n})

NDCG: Normalized Discounted Cumulative Gain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NDCG@k measures the relevance of the ranked recommendations discounted by the rank at which they appear.
It is normalized to be between 0 and 1.
Improving the highest-ranked recommendations has a more important effect than improving the lowest-ranked recommendations.

.. math::
    NDCG@k = \frac{1}{\left | A \right |} \sum_{i=1}^{\left | A \right |} \frac {\sum_{r=1}^{\left | P_i \right |} \frac{rel(P_{i,r})}{log_2(r+1)}}{\sum_{r=1}^{\left | A_i \right |} \frac{1}{log_2(r+1)}}

Diversity Recommender Metrics
-----------------------------
Diversity recommender metrics evaluate the quality of recommendations for different notions of diversity.

Inter-List Diversity
^^^^^^^^^^^^^^^^^^^^
Inter-List Diversity@k measures the inter-list diversity of the recommendations when only k recommendations are
made to the user. It measures how user's lists of recommendations are different from each other. This metric has a range
in :math:`[0, 1]`. The higher this metric is, the more diversified lists of items are recommended to different users.
Let :math:`U` denote the set of :math:`N` unique users, :math:`u_i`, :math:`u_j \in U` denote the i-th and j-th user in the
user set, :math:`i, j \in \{0,1,\cdots,N\}`. :math:`R_{u_i}` is the binary indicator vector representing provided
recommendations for :math:`u_i`. :math:`I` is the set of all unique user pairs, :math:`\forall~i<j, \{u_i, u_j\} \in I`.

.. math::
        Inter\mbox{-}list~diversity = \frac{\sum_{i,j, \{u_i, u_j\} \in I}(cosine\_distance(R_{u_i}, R_{u_j}))}{|I|}

