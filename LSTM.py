import torch
import torch.nn as nn
from utils import get_device
from global_variables import *
device = get_device()


class BiRNN(nn.Module):
    """
    Bidirectional RNN/LSTM followed by three fully connected linear layers
    """

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=11, vocab_size=101,
                 max_len=32, num_layers=2, rnn_type='LSTM', dropout=0.125):
        super(BiRNN, self).__init__()
        self.hidden = None
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define embedding; arguments: num_embeddings, embedding_dim
        self.embedding = nn.Embedding(vocab_size, max_len, padding_idx=vocab_size - 1)

        # Define the LSTM layer
        self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers,
                                           batch_first=True, bidirectional=bidirectional, dropout=dropout)

        # Define the output layer
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 8)
        self.linear3 = nn.Linear(8, 2)

        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool1d(32)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim))

    def forward(self, input):
        
        input = self.embedding(input)
        
        # Creating PackedSequence
        #packing = nn.utils.rnn.pad_sequence(input)
        #                                                  !!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Insert packing stuff
        
        self.hidden = self.init_hidden()

        
        #print(input.dtype)
        #self.hidden = self.init_hidden()
        #print(self.hidden[0].dtype)
        
        #print(input.shape)
        #print(self.hidden[0].shape)
        #print(self.hidden[1].shape)
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size ,hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (batch_size, num_layers, hidden_dim).
        input, self.hidden = self.lstm(input, self.hidden)
        #print(self.hidden[0].shape)
        #print(input.shape)
        
        # Unpacking PackedSequence
        #                                                  !!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Insert packing stuff
        
        
        # Something with Permute???????                    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        input = torch.permute(input, (0,2,1))
        
        # Tanh
        input = self.tanh(input)
        
        # MaxPool 1D
        input = self.maxpool(input)
        #print(input.shape)
        
        # Tanh
        input = self.tanh(input)
        #print(input.shape)
        
        # Squeeze
        input = torch.squeeze(input)
        #print(input.shape)
        
        # First linear layer
        input = self.linear1(input)
        
        # Dropout
        input = self.dropout(input)
        
        # Second linear layer
        input = self.linear2(input)
        
        # Dropout
        input = self.dropout(input)
        
        # Third linear layer
        output = self.linear3(input)
        
        return output
