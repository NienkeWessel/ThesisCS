from MLModel import MLModel
from train import train_onesplit
from LSTM import BiRnn

class PytorchModel(MLModel):
    def __init__(self) -> None:
        pass

    def train(self, X, y):
        train_onesplit(self.model, X, y)

    def predict(self, X):
        pass

class LSTMModel(PytorchModel):
    def __init__(self) -> None:
        lstm_input_size = 32
        hidden_state_size = 256
        batch_size = 64
        num_sequence_layers = 2
        output_dim = 2                       # !!!!!!!!!!!!!!!!!!!!!!!!
        rnn_type = 'LSTM'
        self.model = Bi_RNN(lstm_input_size, hidden_state_size, batch_size=batch_size, output_dim=output_dim, num_layers=num_sequence_layers, rnn_type=rnn_type)