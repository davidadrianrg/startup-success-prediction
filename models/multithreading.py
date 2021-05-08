"""Multithreading module with different Thread classes to implement training models and DNN with cross validation."""

from threading import Thread
from sklearn.model_selection import cross_validate

class ModelThread(Thread):
    """Thread class to implement the training model using cross validation."""

    def __init__(self, X_train, t_train, model, cv, scoring, **kwargs):
        super().__init__(**kwargs)
        self.X_train = X_train
        self.t_train = t_train
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.scores = None

    
    def run(self):
        self.cv_multithread_model()

    def cv_multithread_model(self):
        # Using n_jobs = -1 will use all the cores available in a multithreading process
        self.scores = cross_validate(
                self.model,
                self.X_train,
                self.t_train,
                cv=self.cv,
                scoring=self.scoring,
                return_train_score=True,
                n_jobs=-1,
            )

class DnnThread(Thread):
    """Thread class to implement the training dnn fitting."""

    def __init__(self, X_train, t_train, X_val, t_val, model, epochs, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.X_train = X_train
        self.t_train = t_train
        self.X_val = X_val
        self.t_val = t_val
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = None

    
    def run(self):
        self.fit_dnn()

    def fit_dnn(self):
        # Training of the model
        self.history = self.model.fit(
            self.X_train,
            self.t_train,
            validation_data=(self.X_val, self.t_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )