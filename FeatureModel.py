from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

import pickle

import matplotlib.pyplot as plt

import skops.io as sio

from MLModel import MLModel


class FeatureModel(MLModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        ngrams = params['data_params']['ngrams']
        ngram_range = params['data_params']['ngram_range']

        if ngrams:
            self.vectorizer = CountVectorizer(analyzer='char', lowercase=False, ngram_range=ngram_range, min_df=0.001)
        else: 
            self.vectorizer = None


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
        if not (self.vectorizer is None):
            with open(filename + '_vectorizer.pk', 'wb') as fin:
                pickle.dump(self.vectorizer, fin)

    def load_model(self, filename):
        unknown_types = sio.get_untrusted_types(file=filename)
        self.model = sio.load(filename, trusted=unknown_types)
        if self.params['data_params']['ngrams']:
            with open(filename + '_vectorizer.pk', 'rb') as fin:
                self.vectorizer = pickle.load(fin)


class RandomForest(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        model_params = self.params['model_params']
        if 'n_estimators' in model_params:
            n_estimators = model_params['n_estimators']
        else:
            n_estimators = 100
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

        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            random_state=0)

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
            splitter = "best"

        self.model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                 class_weight=class_weight)

    def __str__(self) -> str:
        return "DecisionTreeModel"
    
    def plot_tree(self, savefile):
        plt.figure(figsize=(20,20))
        tree.plot_tree(self.model, node_ids=True, fontsize=15)
        plt.savefig(f'{savefile}.png', bbox_inches='tight')


class KNearestNeighbors(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)

        model_params = self.params['model_params']
        if 'n_neighbors' in model_params:
            n_neighbors = model_params['n_neighbors']
        else:
            n_neighbors = 5

        if 'weights' in model_params:
            weights = model_params['weights']
        else:
            weights = 'uniform'

        if 'metric' in model_params:
            self.metric = model_params['metric']
        else:
            self.metric = 'minkowski'

        if 'p' in model_params:
            p = model_params['p']
        else:
            p = 2

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p, metric=self.metric)

    def __str__(self) -> str:
        return "KNearestNeighborsModel"

class GaussianNaiveBayes(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.model = GaussianNB()

    def __str__(self) -> str:
        return "GaussianNBModel"


class MultinomialNaiveBayes(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        model_params = self.params['model_params']
        if 'alpha' in model_params:
            alpha = model_params['alpha']
        else:
            alpha = 1.0

        self.model = MultinomialNB(alpha=alpha)

    def __str__(self) -> str:
        return "MultinomialNBModel"

class AdaBoost(FeatureModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        model_params = self.params['model_params']
        if 'n_estimators' in model_params:
            n_estimators = model_params['n_estimators']
        else:
            n_estimators = 50
        if 'learning_rate' in model_params:
            learning_rate = model_params['learning_rate']
        else:
            learning_rate = 1.0

        self.model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

    def __str__(self) -> str:
        return "AdaBoostModel"