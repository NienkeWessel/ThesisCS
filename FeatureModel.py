from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import skops.io as sio

from MLModel import MLModel



class FeatureModel(MLModel):
    def __init__(self) -> None:
        super().__init__()

    def train(self, X, y):
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
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestClassifier(max_depth=10, random_state=0)


class DecisionTree(FeatureModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = tree.DecisionTreeClassifier()


class GaussianNaiveBayes(FeatureModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = GaussianNB()


class MultinomialNaiveBayes(FeatureModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = MultinomialNB()
