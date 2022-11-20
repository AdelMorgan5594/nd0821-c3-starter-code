from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model




def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def compute_model_performance_slice(model, X, categorical_features, label, encoder, lb):
    """ Compute model performance on a slice from the data.
    Inputs
    ------
    model :Trained machine learning model that would run on the data.
    X : Data used for prediction.
    categorical_features : list[str]
        List containing the names of the categorical features.
    Returns
    -------
    performance_slice : the performance o each slice.
    """
    performance_slice = {}
    for feature in categorical_features:
        for feature_value in X[feature].unique().tolist():
            X_slice,y_slice,_,_ = process_data(X[X[feature] == feature_value], categorical_features, label=label, training=False, encoder=encoder, lb=lb)
            preds = inference(model, X_slice)
            performance_slice[feature + '_' + feature_value] = compute_model_metrics(y_slice, preds)

    all_performance_sliced = pd.DataFrame(performance_slice).T
    all_performance_sliced.columns = ['precision', 'recall', 'fbeta']
    all_performance_sliced['slice'] = all_performance_sliced.index
    all_performance_sliced.reset_index(drop=True, inplace=True)

    return all_performance_sliced




def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds