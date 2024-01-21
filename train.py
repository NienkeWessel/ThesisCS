from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l

import numpy as np

from build_datasets import CharacterDataset

from global_variables import *


def train(net, train_input, train_target, val_input, val_target, epochs=100, lr=0.01, device=d2l.try_gpu(), optim="SGD",
          charset='old'):
    """Trains the neural network net for a given train and test split
    
    Keyword arguments:
    net -- the neural network that needs to be trained
    train_input -- words in the training part of the dataset
    train_target -- labels in the training part
    val_input -- words in the validation part
    val_target -- labels in the validation dataset
    epochs -- number of training epochs (default 100)
    lr -- learning rate of the network (default 0.01)
    device -- CPU or GPU (default tries gpu with d2l)
    optim -- optimizer to be used (default SGD)
    charset -- encoding charset to be used (default old charset) New charset does not work properly yet
    """

    if charset == 'old':
        train = CharacterDataset(train_input, y=train_target)
        valid = CharacterDataset(val_input, y=val_target)
    elif charset == 'new':
        train = CharacterDatasetOneHot(train_input, train_target)
        valid = CharacterDatasetOneHot(val_input, val_target)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False, drop_last=True)
    data_loaders = {'train': train_loader,
                    'val': valid_loader}

    net = net.to(device)

    if optim == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    animator = d2l.Animator(xlabel='epoch',
                            legend=['train loss', 'train acc', 'validation acc'],
                            figsize=(10, 5))

    timer = {'train': d2l.Timer(), 'val': d2l.Timer()}

    # Trains the model net with data from the data_loaders['train'] and data_loaders['val'].
    for epoch in range(epochs):
        # monitor loss, accuracy, number of samples
        metrics = {'train': d2l.Accumulator(3), 'val': d2l.Accumulator(3)}

        for phase in ('train', 'val'):
            # switch network to train/eval mode
            net.train(phase == 'train')

            for i, (x, y) in enumerate(data_loaders[phase]):
                timer[phase].start()

                # move to device
                x = x.to(device)
                y = y.to(device)

                # compute prediction
                y_hat = net(x)

                # compute cross-entropy loss
                loss = torch.nn.CrossEntropyLoss()(y_hat, y)

                if phase == 'train':
                    # compute gradients and update weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                metrics[phase].add(loss * x.shape[0],
                                   accuracy(y_hat, y) * x.shape[0],
                                   x.shape[0])

                timer[phase].stop()

        animator.add(epoch + 1,
                     (metrics['train'][0] / metrics['train'][2],
                      metrics['train'][1] / metrics['train'][2],
                      metrics['val'][1] / metrics['val'][2]))

    train_loss = metrics['train'][0] / metrics['train'][2]
    train_acc = metrics['train'][1] / metrics['train'][2]
    val_acc = metrics['val'][1] / metrics['val'][2]
    examples_per_sec = metrics['train'][2] * epochs / timer['train'].sum()

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'val acc {val_acc:.3f}')
    print(f'{examples_per_sec:.1f} examples/sec '
          f'on {str(device)}')


def train_kfold(net, words_train, y_train, epochs=100, lr=0.01, device=d2l.try_gpu(), n_folds=5, charset='old'):
    """Trains the neural network net for k splits
    
    Keyword arguments:
    net -- the neural network that needs to be trained
    words_train -- the dataset of words
    y_train -- the labels of the dataset size 2 x len(words)
    epochs -- number of training epochs (default 100)
    lr -- learning rate of the network (default 0.01)
    device -- CPU or GPU (default tries gpu with d2l)
    n_folds -- the number of folds (default 5)
    charset -- encoding charset to be used (default old charset) New charset does not work properly yet
    """
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    local_val_score = 0
    models = {}

    nr_classes = 2

    # print(len(words_train))
    # print(y_train.shape)

    k = 0  # initialize fold number
    for tr_idx, val_idx in kfold.split(words_train, y_train):
        print('starting fold', k)
        k += 1

        print(6 * '#', 'splitting and reshaping the data')
        words_train = np.array(words_train).flatten()
        train_input = words_train[tr_idx]
        train_target = y_train[tr_idx]
        val_input = words_train[val_idx]
        val_target = y_train[val_idx]

        train(net, train_input, train_target, val_input, val_target, epochs=epochs, lr=lr, device=device,
              charset=charset)


def train_onesplit(net, words, y, train_size=0.8, epochs=100, lr=0.01, device=d2l.try_gpu(), charset='old'):
    """Trains the neural network net for one train-test split
    
    Keyword arguments:
    net -- the neural network that needs to be trained
    words -- the dataset of words
    y -- the labels of the dataset size 2 x len(words)
    train_size -- relative size of trainingset compared to testset (default 0.8)
    epochs -- number of training epochs (default 100)
    lr -- learning rate of the network (default 0.01)
    device -- CPU or GPU (default tries gpu with d2l)
    charset -- encoding charset to be used (default old charset) New charset does not work properly yet
    """
    train_input, val_input, train_target, val_target = train_test_split(words, y, train_size=train_size)
    train(net, train_input, train_target, val_input, val_target, epochs=epochs, lr=lr, device=device, charset=charset)


def accuracy(y_hat, y):
    # Computes the mean accuracy.
    # y_hat: raw network output (before sigmoid or softmax)
    #        shape (samples, classes)
    # y:     shape (samples)
    """if y_hat.shape[1] == 1:
        # binary classification
        y_hat = (y_hat[:, 0] > 0).to(y.dtype)
    else:
        # multi-class classification
        y_hat = torch.argmax(y_hat, axis=1).to(y.dtype)
    """
    y_hat = torch.transpose(
        torch.vstack(((y_hat[:, 0] > y_hat[:, 1]).unsqueeze(0), (y_hat[:, 0] <= y_hat[:, 1]).unsqueeze(0))), 0, 1)
    y_hat = (y_hat >= 0.5).to(y.dtype)

    correct = (y_hat == y).to(torch.float32)
    return torch.mean(correct)
