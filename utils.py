from os import path, listdir
from torch import torch
from tqdm import tqdm
from glob import glob
from d2l import torch as d2l


def get_device():
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    device = d2l.try_gpu()  # NOTE: for now, we fix this to CPU to allow multiprocessing using cpu threads
    return device


def chunkify(input_file, config):
    # based on: https://www.blopig.com/blog/2016/08/processing-large-files-using-python/
    for filename in tqdm(glob(input_file, recursive=True), desc='Chunkify', mininterval=0.1, unit='files',
                         disable=not config.get('progress')):
        if not path.isfile(filename):
            continue
        file_end = path.getsize(filename)
        with open(filename, 'br') as f:
            chunk_end = f.tell()
            while True:
                chunk_start = chunk_end
                f.seek(config.get('chunk size'), 1)
                f.readline()
                chunk_end = f.tell()
                yield chunk_start, chunk_end - chunk_start, filename
                if chunk_end > file_end:
                    break


def find_files_in_folder(dir):
    """
    Lists all files in a directory
    :param dir: the directory of which the files need to be listed
    :return: list of files in that directory
    """
    return listdir(dir)


def confusion(pred, y):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)

    Code from https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    """
    confusion_vector = pred / y
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1    where prediction and truth are 1 (True Positive)
    #   inf  where prediction is 1 and truth is 0 (False Positive)
    #   nan  where prediction and truth are 0(True Negative)
    #   0    where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def pad_data(data):
    # Check if dataset needs to be padded
    batch_size = 64

    # Calculate how many samples you need to pad
    remainder = len(data) % batch_size
    padding_size = batch_size - remainder

    print(len(data))

    # If padding is needed, duplicate samples from the original dataset to pad it
    if padding_size > 0:
        data = data + data[:padding_size]
        

    print(len(data))
    return data

def split_column_title(title):
    """
    Splits a column title into four parts 
    :param title of the column that needs to be split
    :return: four-tuple with (model_type, complete model name, dataset tested on, tag)
    """
    return title.split("-")


grids = {
    "DecisionTree": {
        'criterion': ('gini', 'entropy', 'log_loss'),
        'splitter': ('best', 'random'),
        'max_depth': [5, 10, 50, 100, 200, 500, None],
        'min_samples_split': [2, 5, 10, 50, 100, 200, 500],
        'min_samples_leaf': [1, 5, 10, 50, 100],
        'class_weight': ('balanced', {0: 1, 1: 5}, {0: 1, 1: 10})},
    "RandomForest": {
        'n_estimators': [10, 50, 100, 200],
        'criterion': ('gini', 'entropy', 'log_loss'),
        'max_depth': [5, 10, 50, 100, 200, 500, None],
        'min_samples_split': [2, 5, 10, 50, 100, 200, 500],
        'min_samples_leaf': [1, 5, 10, 50, 100],
    },
    "KNearestNeighbors": {
        'n_neighbors': [2, 3, 5, 10, 20, 50],
        'weights': ('uniform', 'distance'),
        'metric': ('minkowski', 'cosine'),
        'p': [1, 2, 5, 10],
    },
    'MultinomialNaiveBayes': {
        'alpha': [0.0, 0.5, 1.0],  # no smoothing, Lidstone smoothing, and Laplace smoothing
    },
    'AdaBoost': {
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.5, 1.0, 5.0],
    }
}
