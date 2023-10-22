from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from MLModel import MLModel

class FeatureModel(MLModel):
    def __init__(self) -> None:
        pass

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def calc_accuracy(self, y, pred):
        return accuracy_score(y, pred)




class RandomForest(FeatureModel):
    def __init__(self) -> None:
        self.model = RandomForestClassifier(max_depth = 10, random_state=0)


class DecisionTree(FeatureModel):
    def __init__(self) -> None:
        self.model = tree.DecisionTreeClassifier()


class GaussianNaiveBayes(FeatureModel):
    def __init__(self) -> None:
        self.model = GaussianNB()

class MultinomialNaiveBayes(FeatureModel):
    def __init__(self) -> None:
        self.model = MultinomialNB()