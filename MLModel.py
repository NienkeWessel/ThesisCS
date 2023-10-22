from abc import ABC, abstractmethod

class MLModel(ABC):
    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    #@abstractmethod
    #def 
    
    @abstractmethod
    def calc_accuracy(self, y, pred):
        pass