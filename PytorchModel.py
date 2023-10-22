from MLModel import MLModel
from train import train_onesplit
from LSTM import BiRNN
import torch


class PytorchModel(MLModel):
    def __init__(self) -> None:
        pass

    def train(self, X, y):
        train_onesplit(self.model, X, y, epochs=2)

    def predict(self, X):
        pass

    def calc_accuracy(self, y, pred):
        y_hat = torch.transpose(
            torch.vstack(((pred[:, 0] > pred[:, 1]).unsqueeze(0), (pred[:, 0] <= pred[:, 1]).unsqueeze(0))), 0, 1)
        y_hat = (y_hat >= 0.5).to(y.dtype)

        correct = (y_hat == y).to(torch.float32)
        return torch.mean(correct)


class LSTMModel(PytorchModel):
    def __init__(self) -> None:
        lstm_input_size = 32
        hidden_state_size = 256
        batch_size = 64
        num_sequence_layers = 2
        output_dim = 2  # !!!!!!!!!!!!!!!!!!!!!!!!
        rnn_type = 'LSTM'
        self.model = BiRNN(lstm_input_size, hidden_state_size, batch_size=batch_size, output_dim=output_dim,
                           num_layers=num_sequence_layers, rnn_type=rnn_type)
