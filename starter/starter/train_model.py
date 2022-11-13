# Script to train machine learning model.

from ast import dump
import sys
from sklearn.model_selection import train_test_split
from ml.model import train_model, compute_model_metrics, inference, compute_model_performance_slice
from ml.data import process_data, load_data
import os
import pandas as pd
import logging

file_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.insert(0, file_dir)

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
# Add code to load in the data.
logging.info("Loading data...")
data = pd.read_csv('starter/data/census_clean.csv')


# Optional enhancement, use K-fold cross validation instead of a train-test split.
logging.info("splitting the data...")
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
logging.info('process the training data...')
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
logging.info('process the testing data...')
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
logging.info('Training the model...')
model = train_model(X_train, y_train)

logging.info('Saving the model...')
dump(model,file_dir + '/model.pkl')

logging.info('Saving the encoder and lb...')
dump(encoder,file_dir + '/encoder.pkl')
dump(lb,file_dir + '/lb.pkl')

logging.info('Computing the model metric...')
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info('computing the model performance on slices...')
performance = compute_model_performance_slice(model, test, y_test)