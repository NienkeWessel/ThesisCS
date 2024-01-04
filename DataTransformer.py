from abc import ABC, abstractmethod
import numpy as np
import torch.nn.functional
from Levenshtein import distance as lev
from sklearn.feature_extraction.text import CountVectorizer
from transformers import RobertaTokenizerFast
from datasets import Dataset



class DataTransformer(ABC):
    def __init__(self, dataset, params) -> None:
        """
        Class expects words and labels as a list of strings and a list of numbers respectively
        Transforms the data into the format necessary for the relevant ML algorithm
        The length of words and labels should be the same

        :param words: the words dataset as a list of strings
        :param labels: the correct labels as a list of 0s and 1s
        """
        assert (len(dataset['text']) == len(dataset['label']))
        self.params = params

        self.X = dataset['text']
        self.y = dataset['label']


class FeatureDataTransformer(DataTransformer):
    def __init__(self, dataset, params, comparison_pw) -> None:
        """ - base_features: simple feature set such as
        the length of the word and amount of characters from different character sets - levenshtein: calculate the
        Levenshtein distance to most common passwords - ngrams: ngram of characters as features - ngram_range:
        if ngrams is true, this setting determines which ngrams are to be taken into account
        """
        super().__init__(dataset, params)

        levenshtein = params['levenshtein']
        ngrams = params['ngrams']
        ngram_range = params['ngram_range']

        features = [self.counts(word, comparison_pw, levenshtein=levenshtein) for word in dataset['text']]

        if ngrams:
            vectorizer = CountVectorizer(analyzer='char', lowercase=False, ngram_range=ngram_range, min_df=10)
            ngram_features = vectorizer.fit_transform(
                dataset[
                    'text'])  # CountVectorizer returns a sparse matrix. This needs to be converted into a dense matrix in order to be able to concatenate it.
            features = np.concatenate((np.array(features), ngram_features.toarray()), axis=1)  # link features and words

        #total = np.array(list(zip(dataset['text'], features)), dtype=object)

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
    def __init__(self, dataset, params) -> None:
        super().__init__(dataset, params)
        self.X = np.array(self.X).flatten()
        self.y = (torch.nn.functional.one_hot(torch.as_tensor(dataset['label']).to(torch.int64), num_classes=2)).to(
            float)


class PassGPT10Transformer(DataTransformer):
    def __init__(self, dataset, params) -> None:
        super().__init__(dataset, params)
        self.y = (torch.nn.functional.one_hot(torch.as_tensor(dataset['label']).to(torch.int64), num_classes=2)).to(
            float)
        # torch.transpose(torch.as_tensor(labels).to(torch.int64))

        if params['internet']:
            model_loc = "javirandor/passgpt-10characters"
        else:
            model_loc = "passgpt-10characters"

        tokenizer = RobertaTokenizerFast.from_pretrained(model_loc,
                                                         max_len=12, padding="max_length",
                                                         truncation=True, do_lower_case=False,
                                                         strip_accents=False, mask_token="<mask>",
                                                         unk_token="<unk>", pad_token="<pad>",
                                                         truncation_side="right", is_split_into_words=True)

        self.X = tokenizer(dataset['text'], truncation=True, padding=True, max_length=12)  # return_tensors='pt'
        print(self.X)

class ReformerDataTransformer(DataTransformer):
    def __init__(self, dataset, params) -> None:
        super().__init__(dataset, params)
        self.y = (torch.nn.functional.one_hot(torch.as_tensor(dataset['label']).to(torch.int64), num_classes=2)).to(
            float)
        encoded_data = self.encode(dataset['text'])
        print(encoded_data)
        self.X = {
                    'input_ids': encoded_data[0],
                    'attention_mask': encoded_data[1]
        }
        # transform the data structure so that the code that follows will work (i.e. transform into huggingface dataset so that the update in the other file can happen)
        print(self.X)
    
    def encode(self, list_of_strings, pad_token_id=0):
        '''
        Model does not need a tokenizer, but instead uses fixed encoding and decoding
        Code taken from: https://huggingface.co/google/reformer-enwik8
        '''
        max_length = max([len(string) for string in list_of_strings])

        # create emtpy tensors
        attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
        input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

        for idx, string in enumerate(list_of_strings):
            # make sure string is in byte format
            if not isinstance(string, bytes):
                string = str.encode(string)

            input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
            attention_masks[idx, :len(string)] = 1

        return input_ids, attention_masks
        
    # Decoding
    def decode(self, outputs_ids):
        '''
        Model does not need a tokenizer, but instead uses fixed encoding and decoding
        Code taken from: https://huggingface.co/google/reformer-enwik8
        '''
        decoded_outputs = []
        for output_ids in outputs_ids.tolist():
            # transform id back to char IDs < 2 are simply transformed to ""
            decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
        return decoded_outputs