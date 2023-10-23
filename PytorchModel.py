from MLModel import MLModel
from train import train_onesplit
from LSTM import BiRNN
from build_datasets import CharacterDataset


import torch
from d2l import torch as d2l



class PytorchModel(MLModel):
    def __init__(self) -> None:
        pass

    def train(self, X, y):
        train_onesplit(self.model, X, y, epochs=1)

    def predict(self, X):
        device=d2l.try_gpu()

        test = CharacterDataset(X)
        train_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.model.train(False)

        results = None
        for i, x in enumerate(train_loader):
            x = x.to(device)
            temp_res = self.model(x)
            if results is None:
                results = temp_res
            else: 
                results = torch.cat((results, temp_res), 0)
        return results

        

    def calc_accuracy(self, y, pred):
        if len(y) != len(pred):
            nr_batches = int(len(y)/self.batch_size)
            y = y[:self.batch_size*nr_batches]
        y_hat = torch.transpose(
            torch.vstack(((pred[:, 0] > pred[:, 1]).unsqueeze(0), (pred[:, 0] <= pred[:, 1]).unsqueeze(0))), 0, 1)
        y_hat = (y_hat >= 0.5).to(y.dtype)

        correct = (y_hat == y).to(torch.float32)
        return torch.mean(correct)


class LSTMModel(PytorchModel):
    def __init__(self) -> None:
        lstm_input_size = 32
        hidden_state_size = 256
        self.batch_size = 64
        num_sequence_layers = 2
        output_dim = 2  # !!!!!!!!!!!!!!!!!!!!!!!!
        rnn_type = 'LSTM'
        self.model = BiRNN(lstm_input_size, hidden_state_size, batch_size=self.batch_size, output_dim=output_dim,
                           num_layers=num_sequence_layers, rnn_type=rnn_type)
