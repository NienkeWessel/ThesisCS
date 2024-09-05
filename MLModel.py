from abc import ABC, abstractmethod


class MLModel(ABC):
    def __init__(self, params) -> None:
        self.model = None
        self.params = params
        if 'model_loc' in self.params:
            print("A model location is provided in the params")
            self.model_loc = self.params['model_loc']
        else:
            self.model_loc = None

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def train(self, X, y, params=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def calc_accuracy(self, y, pred):
        pass

    @abstractmethod
    def calc_recall(self, y, pred):
        pass

    @abstractmethod
    def calc_precision(self, y, pred):
        pass

    @abstractmethod
    def calc_f1score(self, y, pred):
        pass

    @abstractmethod
    def save_model(self, filename):
        pass

    @abstractmethod
    def load_model(self, filename):
        pass
