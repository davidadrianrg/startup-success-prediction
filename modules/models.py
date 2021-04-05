"""
Training and Evaluating Machine Learning models.

This module implements different functions in order to make it easier to 
train and evaluate the different models variating their hiperparameters
"""

# Importing the required modules
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow import keras
from keras.utils import to_categorical


def normalize_cv_models(model_dict: dict, X: np.ndarray, t: np.ndarray, scoring: list, CV: int = 10,) -> list:
    """Return a list with the scoring tested with cross validation for each model of models_dict."""
    test_sc = []
    for model in model_dict.values():
        model = make_pipeline(StandardScaler(), model)
        scores = cross_validate(model, X, t, cv=CV, scoring=scoring)
        test_sc.append(scores)
    return test_sc

def nn_model(n_inputs: int, neurons_list: list, ncapas: int, learning_rate: float, metrics):
    """Return a deep network model using Adam optimizer."""
    # Defining the layers of the net
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(neurons_list[0], input_dim=n_inputs, activation='relu'))
    for i in range(1,ncapas-1):
        model.add(keras.layers.Dense(neurons_list[i], activation='relu'))
    model.add(keras.layers.Dense(neurons_list[ncapas], activation='softmax'))            
    # After that, we select the optimizer and hyperparameters to use
    opt = keras.optimizers.Adam(learning_rate=learning_rate) 
    # Defining characteristics for the training process
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)    
    return model

def dnn_cv_fit(model: keras.models.Sequential,X:np.ndarray, t:np.ndarray, k_folds: int, k_fold_reps: int, epochs: int, batch_size: int, metrics) -> pd.DataFrame:
    """Train a dnn model using cross validation with stratified kfold and returns a Results Pandas Datafram per each training."""
    # Creating a pandas Dataframe to save the results of the evaluation metrics
    results = pd.DataFrame(columns=metrics)  

    # Generet a stratified K-fold
    rkf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=k_fold_reps, random_state=42) 

    # Make as trainings as values indicated in the cross validation
    for i, (train_index, test_index) in enumerate(rkf.split(X, t)):
        # Shows the step of the k-fold in which we are
        print('k_fold', i+1, 'of', k_folds*k_fold_reps)

        #Obtains the train and test datasets in order to the random index generated in the k-fold
        X_train, t_train = X[train_index], t[train_index]
        X_test, t_test = X[test_index], t[test_index]

        # Recode the classes in as many binary values as classes
        n_classes = len(np.unique(t))
        t_train = to_categorical(t_train, num_classes=n_classes)
        t_test = to_categorical(t_test, num_classes=n_classes)    

        # Train the DNN
        model.fit(X_train, t_train, validation_data=(X_test, t_test), epochs=epochs, batch_size=batch_size, verbose=0)

        # Add a line in the pandas results dataframe with the results of the train
        results.loc[i] = model.evaluate(X_test, t_test, batch_size=None)[1:]  # Discard metric 0, is the value of the loss function

    return results