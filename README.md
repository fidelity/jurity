# Jurity: Fairness & Evaluation Library

Jurity is a research library that provides classification metrics, fairness metrics, recommender system evaluations, and bias mitigation techniques. The library adheres to PEP-8 standards and is tested heavily.

Jurity is developed by the Artificial Intelligence Center of Excellence at Fidelity Investments.

## Fairness Metrics
* [Average Odds](https://fidelity.github.io/jurity/about.html#average-odds)
* [Disparate Impact](https://fidelity.github.io/jurity/about.html#disparate-impact)
* [Equal Opportunity](https://fidelity.github.io/jurity/about.html#equal-opportunity)
* [False Negative Rate (FNR) Difference](https://fidelity.github.io/jurity/about.html#fnr-difference)
* [Generalized Entropy Index](https://fidelity.github.io/jurity/about.html#generalized-entropy-index)
* [Predictive Equality](https://fidelity.github.io/jurity/about.html#predictive-equality)
* [Statistical Parity](https://fidelity.github.io/jurity/about.html#statistical-parity)
* [Theil Index](https://fidelity.github.io/jurity/about.html#theil-index)

## Binary Bias Mitigation Techniques
* [Equalized Odds](https://fidelity.github.io/jurity/about.html#equalized-odds)

## Recommenders Metrics
* [CTR: Click-through rate](https://fidelity.github.io/jurity/about.html#ctr-click-through-rate)
* [Precision@K](https://fidelity.github.io/jurity/about.html#precision)
* [Recall@K](https://fidelity.github.io/jurity/about.html#recall)
* [MAP@K: Mean Average Precision](https://fidelity.github.io/jurity/about.html#map-mean-average-precision)
* [NDCG: Normalized discounted cumulative gain](https://fidelity.github.io/jurity/about.html#ndcg-normalized-discounted-cumulative-gain)

## Classification Metrics
* [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
* [F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
* [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)


## Quick Start: Fairness Evaluation

```python
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
```

## Quick Start: Bias Mitigation

```python
# Import binary fairness and binary bias mitigation
from jurity.mitigation import BinaryMitigation
from jurity.fairness import BinaryFairnessMetrics

# Data
labels = [1, 1, 0, 1, 0, 0, 1, 0]
predictions = [0, 0, 0, 1, 1, 1, 1, 0]
likelihoods = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1]
is_member = [0, 0, 0, 0, 1, 1, 1, 1]

# Bias Mitigation
mitigation = BinaryMitigation.EqualizedOdds()

# Training: Learn mixing rates from the labeled data
mitigation.fit(labels, predictions, likelihoods, is_member)

# Testing: Mitigate bias in predictions
fair_predictions, fair_likelihoods = mitigation.transform(predictions, likelihoods, is_member)

# Scores: Fairness before and after
print("Fairness Metrics Before:", BinaryFairnessMetrics().get_all_scores(labels, predictions, is_member), '\n'+30*'-')
print("Fairness Metrics After:", BinaryFairnessMetrics().get_all_scores(labels, fair_predictions, is_member))
```

## Quick Start: Recommenders Evaluation

```python
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
```

## Quick Start: Classification Evaluation

```python
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
```


## Installation

### Requirements

The library requires Python **3.6+** and depends on standard packages such as ```pandas, numpy``` 
The ``requirements.txt`` lists the necessary packages. 

### Install from wheel package

After installing the requirements, you can install the library from the provided wheel package using the following commands:

```bash
pip install dist/jurity-X.X.X-py3-none-any.whl
```
Note: Don't forget to replace ``X.X.X`` with the current version number.

### Install from source code

Alternatively, you can build a wheel package on your platform from scratch using the source code:

```bash
pip install setuptools wheel # if wheel is not installed
python setup.py bdist_wheel
pip install dist/jurity-X.X.X-py3-none-any.whl
```

### Test Your Setup
To confirm that cloning the repo was successful, run the first example in the [Quick Start](#quick-start-fairness-evaluation). 
To confirm that the whole installation was successful, run the tests and all should pass. 

```bash
python -m unittest discover -v tests
```

### Upgrading the Library

To upgrade to the latest version of the library, run ``git pull origin master`` in the repo folder,
and then run ``pip install --upgrade --no-cache-dir dist/jurity-X.X.X-py3-none-any.whl``.

## Support
Please submit bug reports and feature requests as Issues. You can also submit any additional questions or feedback as issues.

## License
Jurity is licensed under the [Apache License 2.0.](https://github.com/fidelity/jurity/blob/master/LICENSE)
