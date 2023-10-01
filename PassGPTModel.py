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

class PassGPT10Model(GPTModel):
    def __init__(self) -> None:
        self.model = GPT2LMHeadModel.from_pretrained("javirandor/passgpt-10characters").eval()