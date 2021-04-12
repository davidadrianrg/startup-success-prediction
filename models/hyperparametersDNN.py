"""Module with utils functions to optimize hyperparameters and get the best DNN model."""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from keras.utils import to_categorical
from scipy.stats import loguniform


def create_random_network(
    m,
    metrics="accuracy",
    layers=[1, 5],
    n=[10, 20],
    activation=["relu", "sigmoid"],
    lr=[1e-5, 1e-3],
):
    """Return a deep neural network model with pseudo-random hyperparameters according to its args, each time it is called."""
    # Defining the model class Sequential in order to add layers one by one
    model = keras.models.Sequential()
    # First layer of de network
    model.add(
        keras.layers.Dense(
            random.randint(n[0], n[1]),
            input_dim=m,
            activation=random.choice(activation),
        )
    )
    # Input Normalization, working as a StandardScaler() tranformer for input data
    keras.layers.BatchNormalization()
    # Loop which adds layers following a uniform random distribution
    for i in range(random.randint(layers[0], layers[1])):
        model.add(
            keras.layers.Dense(
                random.randint(n[0], n[1]), activation=random.choice(activation)
            )
        )
    model.add(keras.layers.Dense(2, activation="softmax"))
    # Select the Adam optimizer as a general option due to its ability to leave saddle points
    # Choosing a learning rate from a logarithmic uniform distrbution
    optimizer = keras.optimizers.Adam(
        learning_rate=round(loguniform.rvs(lr[0], lr[1]), int(abs(math.log(lr[0], 10))))
    )
    # Define some characteristics for the training process, using categorical crossentropy as the function error
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)
    return model


def get_hyperparams(model):
    """Return the hyperparams of the model passed as an argument."""
    neurons = dict()
    nlayers = len(model.layers)
    for i in range(nlayers):
        neurons["l" + str(i)] = model.layers[i].units
    lr = model.optimizer.learning_rate.value().numpy()
    params = {"nlayers": nlayers, "neurons": neurons, "lr": lr}
    return params


def save_history(history, historial_train, historial_test, metric="accuracy"):
    """Save and return history extension of the metric selected for the model."""
    if "accuracy" not in history.history:
        metric = "loss"
    historial_train.extend(history.history[metric])
    historial_test.extend(history.history["val_" + metric])

    return historial_train, historial_test


def get_best_DNN(
    X,
    t,
    kfolds=5,
    train_size=0.85,
    trials=5,
    epochs=100,
    batch_size=40,
    metrics="accuracy",
):
    """Train the current model using cross validation and register its score comparing with new ones in each trial."""
    # Its needed the accuracy metric, if it is not passed it will be auto-included
    if "accuracy" not in metrics:
        metrics.append("accuracy")

    cv = KFold(kfolds)
    last_mean = 0
    last_trained = 0
    n, m = X.shape
    n_classes = len(np.unique(t))

    # Split the data into train and test sets. Note that test set will be reserved for evaluate the best model later with data unseen previously for it
    X_train_set, X_test, t_train_set, t_test = train_test_split(
        X, t, train_size=train_size
    )

    # Loop for different trials or models to train in order to find the best
    for row in range(trials):

        model = create_random_network(m, metrics)
        params = get_hyperparams(model)

        print(f"\n***Trial {row+1} hyperparameters***", end="\n\n")
        print(params)
        fold = 1
        # Lists to store historical data and means per fold for models
        historial_train = []
        historial_test = []
        mean_folds = []

        # Loop that manage cross validation using training set
        for train_index, test_index in cv.split(X_train_set):
            X_train, X_test = X_train_set[train_index], X_train_set[test_index]
            t_train, t_test = t_train_set[train_index], t_train_set[test_index]

            t_train = to_categorical(t_train, num_classes=n_classes)
            t_test = to_categorical(t_test, num_classes=n_classes)

            # Training of the model
            history = model.fit(
                X_train,
                t_train,
                validation_data=(X_test, t_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )
            # Save the train and test results,of the current model
            historial_train, historial_test = save_history(
                history, historial_train, historial_test
            )

            mean = np.mean(history.history["val_accuracy"])
            mean_folds.append(mean)

            print(f"\nKFold{fold} --- Current score: {mean}")
            fold += 1
        # Criterion to choose the best model with the highest accuracy score in validation
        means = np.mean(mean_folds)
        print(
            f"\n\tMin. Train score: {min(historial_train)} | Max. Train score: {max(historial_train)}"
        )
        if means > last_mean:
            # Save the model and its historial results
            best_model = (model, historial_train, historial_test)
            last_mean = means
    return best_model


def plot_best_DNN(best_model):
    """Plot the validation curve of the model using its historial results."""
    plt.plot(best_model[1])
    plt.plot(best_model[2])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration (epoch)")
    plt.legend(["Trainning", "Test"], loc="lower right")
    plt.show()


"""
Example of code
X = pd.read_csv("X.csv")
X = X.drop(["Unnamed: 0"], axis=1).values
t = pd.read_csv("t.csv")
t = t["labels"].values
n, m = X.shape
n_classes = len(np.unique(t))

best = get_best_DNN(X, t)
plot_best_DNN(best)
"""
