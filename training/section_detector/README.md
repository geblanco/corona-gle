### How to use section classifier?

#### Basic information

Classifier using the Flair library. Use glove inlays to predict the most likely section from the section text.

Currently he predicts the following classes: abstract, conclusions, experiments_or_results, introduction, method, sota.

#### Use

1. First download the weights of the last model: https://drive.google.com/file/d/1ofRFExV_yncyvVoKx5KzXHNzrot-evdO/view?usp=sharing.
2. Use `predict_classifier.py`example to load the model an predict section titles.

#### Results

**Accuracy:** 0.8913

**PRECISION**	**RECALL**	**F1**
0.8913	0.8913	0.8913

**MICRO_AVG**: acc 0.9638 - f1-score 0.8913
**MACRO_AVG**: acc 0.9638 - f1-score 0.8249666666666666

**Classes:**

\#abstract  tp: 1452 - fp: 36 - fn: 144 - tn: 3567 - precision: 0.9758 - recall: 0.9098 - accuracy: 0.9654 - f1-score: 0.9416
\#conclusions tp: 1623 - fp: 172 - fn: 34 - tn: 3370 - precision: 0.9042 - recall: 0.9795 - accuracy: 0.9604 - f1-score: 0.9403
\#experiments_or_results tp: 372 - fp: 39 - fn: 22 - tn: 4766 - precision: 0.9051 - recall: 0.9442 - accuracy: 0.9883 - f1-score: 0.9242
\#introduction tp: 931 - fp: 61 - fn: 289 - tn: 3918 - precision: 0.9385 - recall: 0.7631 - accuracy: 0.9327 - f1-score: 0.8418
\#method    tp: 139 - fp: 19 - fn: 30 - tn: 5011 - precision: 0.8797 - recall: 0.8225 - accuracy: 0.9906 - f1-score: 0.8501
\#sota      tp: 117 - fp: 238 - fn: 46 - tn: 4798 - precision: 0.3296 - recall: 0.7178 - accuracy: 0.9454 - f1-score: 0.4518