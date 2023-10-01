from abc import ABC, abstractmethod
import numpy as np
import torch.nn.functional
from Levenshtein import distance as lev
from sklearn.feature_extraction.text import CountVectorizer


class DataTransformer(ABC):
    def __init__(self, words, labels) -> None:
        """
        Class expects words and labels as a list of strings and a list of numbers respectively
        Transforms the data into the format necessary for the relevant ML algorithm
        The length of words and labels should be the same

        :param words: the words dataset as a list of strings
        :param labels: the correct labels as a list of 0s and 1s
        """
        assert (len(words) == len(labels))

        self.X = words
        self.y = labels


class FeatureDataTransformer(DataTransformer):
    def __init__(self, words, labels, comparison_pw) -> None:
        """ - base_features: simple feature set such as
        the length of the word and amount of characters from different character sets - levenshtein: calculate the
        Levenshtein distance to most common passwords - ngrams: ngram of characters as features - ngram_range:
        if ngrams is true, this setting determines which ngrams are to be taken into account
        """
        super().__init__(words, labels)

        levenshtein = False
        ngrams = False
        ngram_range = (1, 2)

        # features = [counts(pw, levenshtein=levenshtein) for pw in passwords]
        features = [self.counts(word, comparison_pw, levenshtein=levenshtein) for word in words]
        # features = np.concatenate((features, features_word), axis=0)

        if ngrams:
            vectorizer = CountVectorizer(analyzer='char', lowercase=False, ngram_range=ngram_range, min_df=10)
            ngram_features = vectorizer.fit_transform(
                words)  # CountVectorizer returns a sparse matrix. This needs to be converted into a dense matrix in order to be able to concatenate it.
            features = np.concatenate((np.array(features), ngram_features.toarray()), axis=1)  # link features and words

        total = np.array(list(zip(words, features)))

        self.X = features
        self.y = labels

    def counts(self, word, comparison_pw, levenshtein=False):
        alpha_lower = 0
        alpha_upper = 0
        numeric = 0
        special = 0
        s = ""

        for c in word:
            if c.islower():
                alpha_lower += 1
                s += 'L'
            elif c.isupper():
                alpha_upper += 1
                s += 'U'
            elif c.isnumeric():
                numeric += 1
                s += 'N'
            else:
                special += 1
                s += 'S'
        length = len(word)
        char_sets = bool(alpha_lower) + bool(alpha_upper) + bool(numeric) + bool(special)

        if levenshtein:
            lev_d = self.calculate_levenshtein_distance(word, comparison_pw)
            return [length, alpha_lower, alpha_lower / length, alpha_upper, alpha_upper / length, numeric,
                    numeric / length, special, special / length, char_sets, self.count_non_repeating(s), lev_d]
        else:
            return [length, alpha_lower, alpha_lower / length, alpha_upper, alpha_upper / length, numeric,
                    numeric / length, special, special / length, char_sets, self.count_non_repeating(s)]

    def count_non_repeating(self, text):
        """ Remove repeating letters from a string
        E.g. aaabbbccccccaaa becomes abca

        :param text: input text

        return: text without repeating letters """

        count = 0
        for i, c in enumerate(text):
            if i == 0 or c != text[i - 1]:
                count += 1
        return count

    def calculate_levenshtein_distance(self, word, passwords):
        low = 42000
        for pw in passwords:
            d = lev(word, pw, score_cutoff=low - 1)
            if d < low:
                low = d
                if low == 0:
                    return low
        return low


class PytorchDataTransformer(DataTransformer):
    def __init__(self, words, labels) -> None:
        super().__init__(words, labels)
        self.y = (torch.nn.functional.one_hot(torch.as_tensor(labels).to(torch.int64), num_classes=2)).to(float)

class PassGPT10Transformer(DataTransformer):
    def __init__(self, words, labels) -> None:
        self.y = (torch.nn.functional.one_hot(torch.as_tensor(labels).to(torch.int64), num_classes=2)).to(float)

        self.X = None