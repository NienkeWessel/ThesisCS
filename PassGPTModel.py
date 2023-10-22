from MLModel import MLModel
from transformers import GPT2LMHeadModel
from transformers import RobertaTokenizerFast
import torch

class GPTModel(MLModel):
    def __init__(self) -> None:
        pass

    def train(self, X, y):
        pass

    def predict(self, X):
        self.model.forward(X)

    def calc_accuracy(self, y, pred):
        pass

class PassGPT10Model(GPTModel):
    def __init__(self, internet=True) -> None:
        if internet:
            model_loc = "javirandor/passgpt-10characters"
        else:
            model_loc = "passgpt-10characters"
        self.model = GPT2LMHeadModel.from_pretrained(model_loc).eval()