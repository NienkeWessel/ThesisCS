import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import string


class CharacterDataset(Dataset):
    """Custom dataset for character level embedding 
    
    Parameters
    ---------- 
    text -- input text
    y -- labels
    max_len -- the maximum length of a (pass)word
    
    Attributes
    ---------- 
    ch2ix : defaultdict Mapping from the character to the position of that character in the vocabulary. Note that all characters that are not in the vocabulary will get mapped into the index `vocab_size - 1`. 
    ix2ch : dict Mapping from the character position in the vocabulary to the actual character. 
    vocabulary : list List of all characters
    """

    def __init__(self, texts, y=None, max_len=32):
        self.texts = texts
        if y is None:
            self.has_y = False
        else: 
            self.y = y
            self.has_y = True
        self.max_len = max_len
        self.vocabulary = string.printable

        self.ch2ix = {
            x[0]: i
            for i, x in enumerate(self.vocabulary)
        }

        self.ix2ch = {v: k for k, v in self.ch2ix.items()}


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, ix):
        for c in self.texts[ix]:
            if c not in self.ch2ix:
                self.ch2ix[c] = len(self.vocabulary)
        X = torch.LongTensor(
            [self.ch2ix[c] for c in self.texts[ix]]
        )
        if len(X) > self.max_len:
            X = X[:self.max_len]
        X = nn.ConstantPad1d((0, self.max_len - len(X)), 0)(X)
        
        if self.has_y:
            y = self.y[ix]
            return X, y
        else: 
            return X
