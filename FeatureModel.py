from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import skops.io as sio

from MLModel import MLModel


class FeatureModel(MLModel):
    def __init__(self, params) -> None:
        super().__init__(params)

    @abstractmethod
    def __str__(self) -> str:
        pass

    def train(self, X, y, params=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def calc_accuracy(self, y, pred):
        return accuracy_score(y, pred)

    def calc_recall(self, y, pred):
        return recall_score(y, pred)

    def calc_precision(self, y, pred):
        return precision_score(y, pred)

    def calc_f1score(self, y, pred):
        return f1_score(y, pred)

    def save_model(self, filename):
        sio.dump(self.model, filename)

    def load_model(self, filename):
        self.model = sio.load(filename)


class RandomForest(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.model = RandomForestClassifier(max_depth=10, random_state=0)

    def __str__(self) -> str:
        return "RandomForestModel"


class DecisionTree(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)

        model_params = self.params['model_params']
        if 'criterion' in model_params:
            criterion = model_params['criterion']
        else:
            criterion = "gini"

        if 'max_depth' in model_params:
            max_depth = model_params['max_depth']
        else:
            max_depth = None

        if 'min_samples_split' in model_params:
            min_samples_split = model_params['min_samples_split']
        else:
            min_samples_split = 2

        if 'min_samples_leaf' in model_params:
            min_samples_leaf = model_params['min_samples_leaf']
        else:
            min_samples_leaf = 1

        if 'class_weight' in model_params:
            class_weight = model_params['class_weight']
        else:
            class_weight = None

        if 'splitter' in model_params:
            splitter = model_params['splitter']
        else:
            splitter = None

        self.model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                 class_weight=class_weight)

    def __str__(self) -> str:
        return "DecisionTreeModel"


class GaussianNaiveBayes(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.model = GaussianNB()

    def __str__(self) -> str:
        return "GaussianNBModel"


class MultinomialNaiveBayes(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.model = MultinomialNB()

    def __str__(self) -> str:
        return "MultinomialNBModel"
