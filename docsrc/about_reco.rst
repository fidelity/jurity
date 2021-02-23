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

CTR measures the accuracy of the recommendations over the subset of user-item pairs that appear in both actual ratings and recommendations.

Let :math:`M` denote the set of user-item pairs that appear in both actual ratings and recommendations, and :math:`C(M_i)` be an indicator function that produces :math:`1` if the user clicked on the item, and :math:`0` if they didn't.

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
