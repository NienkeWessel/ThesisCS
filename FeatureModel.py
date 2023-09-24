from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier

from MLModel import MLModel

class FeatureModel(MLModel):
    def __init__(self) -> None:
        pass

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)




class RandomForest(FeatureModel):
    def __init__(self) -> None:
        self.model = RandomForestClassifier(max_depth = 10, random_state=0)


        

