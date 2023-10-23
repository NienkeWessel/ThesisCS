from abc import ABC, abstractmethod
import numpy as np
import torch.nn.functional
from Levenshtein import distance as lev
from sklearn.feature_extraction.text import CountVectorizer
from transformers import RobertaTokenizerFast



class DataTransformer(ABC):
    def __init__(self, dataset) -> None:
        """
        Class expects words and labels as a list of strings and a list of numbers respectively
        Transforms the data into the format necessary for the relevant ML algorithm
        The length of words and labels should be the same

        :param words: the words dataset as a list of strings
        :param labels: the correct labels as a list of 0s and 1s
        """
        assert (len(dataset['text']) == len(dataset['label']))

        self.X = dataset['text']
        self.y = dataset['label']


class FeatureDataTransformer(DataTransformer):
    def __init__(self, dataset, comparison_pw) -> None:
        """ - base_features: simple feature set such as
        the length of the word and amount of characters from different character sets - levenshtein: calculate the
        Levenshtein distance to most common passwords - ngrams: ngram of characters as features - ngram_range:
        if ngrams is true, this setting determines which ngrams are to be taken into account
        """
        super().__init__(dataset)

        levenshtein = False
        ngrams = False
        ngram_range = (1, 2)

        # features = [counts(pw, levenshtein=levenshtein) for pw in passwords]
        features = [self.counts(word, comparison_pw, levenshtein=levenshtein) for word in dataset['text']]
        # features = np.concatenate((features, features_word), axis=0)

        if ngrams:
            vectorizer = CountVectorizer(analyzer='char', lowercase=False, ngram_range=ngram_range, min_df=10)
            ngram_features = vectorizer.fit_transform(
                dataset['text'])  # CountVectorizer returns a sparse matrix. This needs to be converted into a dense matrix in order to be able to concatenate it.
            features = np.concatenate((np.array(features), ngram_features.toarray()), axis=1)  # link features and words

        total = np.array(list(zip(dataset['text'], features)))

        self.X = features
        self.y = dataset['label']

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
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.X = np.array(self.X).flatten()
        self.y = (torch.nn.functional.one_hot(torch.as_tensor(dataset['label']).to(torch.int64), num_classes=2)).to(float)

class PassGPT10Transformer(DataTransformer):
    def __init__(self, dataset, internet=True) -> None:
        self.y = (torch.nn.functional.one_hot(torch.as_tensor(dataset['label']).to(torch.int64), num_classes=2)).to(float)
        #torch.transpose(torch.as_tensor(labels).to(torch.int64))

        if internet:
            model_loc = "javirandor/passgpt-10characters"
        else:
            model_loc = "passgpt-10characters"

        tokenizer = RobertaTokenizerFast.from_pretrained(model_loc,
                                                 max_len=12, padding="max_length",
                                                 truncation=True, do_lower_case=False,
                                                 strip_accents=False, mask_token="<mask>",
                                                 unk_token="<unk>", pad_token="<pad>",
                                                 truncation_side="right", is_split_into_words=True)
        

        self.X = tokenizer(dataset['text'], truncation = True, padding = True, max_length=12) #return_tensors='pt'
