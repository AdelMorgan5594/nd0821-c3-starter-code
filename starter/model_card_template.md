# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random forest classifier from the scikit-learn library

## Intended Use
It is an estimator that fits a number of decision tree classifiers on subsets of the data.Then performs averaging to improve the accuracy. 
## Training Data
The data is Census Income Data Set from UCI Machine learning repository - https://archive.ics.uci.edu/ml/datasets/census+income). 80 % of the data is used for training
## Evaluation Data
The data is Census Income Data Set from UCI Machine learning repository - https://archive.ics.uci.edu/ml/datasets/census+income). 20 % of the data is used for Testing
## Metrics
we have used 3 metrics to evaluate the model. percision, recall and f_beta.

## Ethical Considerations
The model is trained on the census data set. The data set is a public data set. The data set is not biased. The model is not biased. The model is not used for any unethical purpose.

## Caveats and Recommendations
We need to update the data set with the latest census data to prevent any data drifts.
