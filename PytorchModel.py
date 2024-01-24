from MLModel import MLModel
from train import train_onesplit
from LSTM import BiRNN
from build_datasets import CharacterDataset
from utils import confusion

import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

device = d2l.try_gpu()


class PytorchModel(MLModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        try:
            self.batch_size = self.params['model_params']['batch_size']
        except KeyError:
            raise KeyError("Not all model parameters have been specified as needed, check if batch_size is "
                           "specified properly")

    def train(self, X, y, params=None):
        if 'epochs' in params:
            epochs = params['epochs']
        else:
            epochs = 1
        if 'fig_name' in params:
            fig_name = params['fig_name']
        else:
            fig_name = "loss.png"
        train_onesplit(self.model, X, y, epochs=epochs)

        # save the plot made in the train function in the train.py file
        plt.savefig(fig_name)

    def predict(self, X):
        test = CharacterDataset(X)
        train_loader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.model.train(False)

        self.model = self.model.to(device)

        results = None
        for i, x in enumerate(train_loader):
            x = x.to(device)
            with torch.no_grad():
                temp_res = self.model(x)
            if results is None:
                results = temp_res
            else:
                results = torch.cat((results, temp_res), 0)
        return results

    def cut_of_y_to_batchsize(self, y):
        nr_batches = int(len(y) / self.batch_size)
        y = y[:self.batch_size * nr_batches]
        return y

    def transform_pred(self, pred, y):
        y_hat = torch.transpose(
            torch.vstack(((pred[:, 0] > pred[:, 1]).unsqueeze(0), (pred[:, 0] <= pred[:, 1]).unsqueeze(0))), 0, 1)
        return (y_hat >= 0.5).to(y.dtype)


    def calc_accuracy(self, y, pred):

        y = y.to(device)
        pred = pred.to(device)

        if len(y) != len(pred):
            y = self.cut_of_y_to_batchsize(y)

        y_hat = torch.transpose(
            torch.vstack(((pred[:, 0] > pred[:, 1]).unsqueeze(0), (pred[:, 0] <= pred[:, 1]).unsqueeze(0))), 0, 1)
        y_hat = (y_hat >= 0.5).to(y.dtype)

        correct = (y_hat == y).to(torch.float32)
        return torch.mean(correct).tolist()

    def calc_recall(self, y, pred):
        y = y.to(device)
        pred = pred.to(device)
        if len(y) != len(pred):
            y = self.cut_of_y_to_batchsize(y)
        pred = self.transform_pred(pred, y)
        tp, fp, tn, fn = confusion(pred, y)
        return tp / (tp + fn)

    def calc_precision(self, y, pred):
        y = y.to(device)
        pred = pred.to(device)
        if len(y) != len(pred):
            y = self.cut_of_y_to_batchsize(y)
        pred = self.transform_pred(pred, y)
        tp, fp, tn, fn = confusion(pred, y)
        return tp / (tp+fp)

    def calc_f1score(self, y, pred):
        y = y.to(device)
        pred = pred.to(device)
        if len(y) != len(pred):
            y = self.cut_of_y_to_batchsize(y)
        pred = self.transform_pred(pred, y)
        tp, fp, tn, fn = confusion(pred, y)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return (2 * precision * recall) / (precision + recall)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=torch.device(device)))

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)


class LSTMModel(PytorchModel):
    def __init__(self, params) -> None:
        super().__init__(params)
        if self.model_loc is not None:
            self.model = self.load_model(self.model_loc)
        else:
            lstm_input_size = 32
            hidden_state_size = 256
            self.batch_size = 64
            num_sequence_layers = 2
            output_dim = 2  # !!!!!!!!!!!!!!!!!!!!!!!!
            self.rnn_type = 'LSTM'
            self.model = BiRNN(lstm_input_size, hidden_state_size, batch_size=self.batch_size, output_dim=output_dim,
                               num_layers=num_sequence_layers, rnn_type=self.rnn_type)

    def __str__(self) -> str:
        return self.rnn_type + "Model"
