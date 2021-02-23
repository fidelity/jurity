.. _quick:

Quick Start 
===========

Calculate Fairness Metrics
--------------------------

.. code-block:: python

    # Import binary and multi-class fairness metrics
    from jurity.fairness import BinaryFairnessMetrics, MultiClassFairnessMetrics

    # Data
    binary_predictions = [1, 1, 0, 1, 0, 0]
    multi_class_predictions = ["a", "b", "c", "b", "a", "a"]
    multi_class_multi_label_predictions = [["a", "b"], ["b", "c"], ["b"], ["a", "b"], ["c", "a"], ["c"]]
    is_member = [0, 0, 0, 1, 1, 1]
    classes = ["a", "b", "c"]

    # Metrics (see also other available metrics)
    metric = BinaryFairnessMetrics.StatisticalParity()
    multi_metric = MultiClassFairnessMetrics.StatisticalParity(classes)

    # Scores
    print("Metric:", metric.description)
    print("Lower Bound: ", metric.lower_bound)
    print("Upper Bound: ", metric.upper_bound)
    print("Ideal Value: ", metric.ideal_value)
    print("Binary Fairness score: ", metric.get_score(binary_predictions, is_member))
    print("Multi-class Fairness scores: ", multi_metric.get_scores(multi_class_predictions, is_member))
    print("Multi-class multi-label Fairness scores: ", multi_metric.get_scores(multi_class_multi_label_predictions, is_member))

Fit and Apply Bias Mitigation
-----------------------------

.. code-block:: python

    # Import binary fairness metrics and mitigation
    from jurity.fairness import BinaryFairnessMetrics
    from jurity.mitigation import BinaryMitigation

    # Data
    labels = [1, 1, 0, 1, 0, 0, 1, 0]
    predictions = [0, 0, 0, 1, 1, 1, 1, 0]
    likelihoods = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1]
    is_member = [0, 0, 0, 0, 1, 1, 1, 1]

    # Bias Mitigation
    mitigation = BinaryMitigation.EqualizedOdds()

    # Training: Learn mixing rates from labeled data
    mitigation.fit(labels, predictions, likelihoods, is_member)

    # Testing: Mitigate bias in predictions
    fair_predictions, fair_likelihoods = mitigation.transform(predictions, likelihoods, is_member)

    # Results: Fairness before and after
    print("Fairness Metrics Before:", BinaryFairnessMetrics().get_all_scores(labels, predictions, is_member), '\n'+30*'-')
    print("Fairness Metrics After:", BinaryFairnessMetrics().get_all_scores(labels, fair_predictions, is_member))
    
Calculate Recommenders Metrics
------------------------------

.. code-block:: python

    # Import recommenders metrics
    from jurity.recommenders import BinaryRecoMetrics, RankingRecoMetrics
    import pandas as pd

    # Data
    actual = pd.DataFrame({"user_id": [1, 2, 3, 4], "item_id": [1, 2, 0, 3], "clicks": [0, 1, 0, 0]})
    predicted = pd.DataFrame({"user_id": [1, 2, 3, 4], "item_id": [1, 2, 2, 3], "clicks": [0.8, 0.7, 0.8, 0.7]})

    # Metrics
    ctr = BinaryRecoMetrics.CTR(click_column="clicks")
    ncdg_k = RankingRecoMetrics.NDCG(click_column="clicks", k=3)
    precision_k = RankingRecoMetrics.Precision(click_column="clicks", k=2)
    recall_k = RankingRecoMetrics.Recall(click_column="clicks", k=2)
    map_k = RankingRecoMetrics.MAP(click_column="clicks", k=2)

    # Scores
    print("CTR:", ctr.get_score(actual, predicted))
    print("NCDG:", ncdg_k.get_score(actual, predicted))
    print("Precision@K:", precision_k.get_score(actual, predicted))
    print("Recall@K:", recall_k.get_score(actual, predicted))
    print("MAP@K:", map_k.get_score(actual, predicted))

Calculate Classification Metrics
--------------------------------

.. code-block:: python
    
    # Import classification metrics
    from jurity.classification import BinaryClassificationMetrics

    # Data
    labels = [1, 1, 0, 1, 0, 0, 1, 0]
    predictions = [0, 0, 0, 1, 1, 1, 1, 0]
    likelihoods = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1]
    is_member = [0, 0, 0, 0, 1, 1, 1, 1]

    # Available: Accuracy, F1, Precision, Recall, and AUC 
    f1_score = BinaryClassificationMetrics.F1()

    print('F1 score is', f1_score.get_score(predictions, labels))