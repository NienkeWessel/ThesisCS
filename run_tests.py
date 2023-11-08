import os
import time
import json
from datasets import load_from_disk, Dataset

from sklearn.model_selection import GridSearchCV

from DataTransformer import FeatureDataTransformer, PytorchDataTransformer, PassGPT10Transformer
from FeatureModel import FeatureModel
from PytorchModel import PytorchModel, LSTMModel
from PassGPTModel import HuggingfaceModel, PassGPT10Model
from FeatureModel import RandomForest, DecisionTree, GaussianNaiveBayes, MultinomialNaiveBayes


def find_files(dir):
    """
    Lists all files in a directory
    :param dir: the directory of which the files need to be listed
    :return: list of files in that directory
    """
    return os.listdir(dir)


def create_dummy_dataset():
    """
    Creates a dummy dataset of the right format, so that you can test some toy data
    :return:
    """
    dataset = Dataset.from_dict({
        'text': ['password', 'normal'],
        'label': [1.0, 0.0]
    })
    return dataset


def transform_data(model, dataset, comparison_pw):
    """
    Transforms the data by using a DataTransformer that fits the model type
    :param model: the model the dataset is for
    :param dataset: the dataset with at least a train, test and validation part
    :param split: the part of the dataset you want; train, test or validation
    :return: the transformed data in a DataTransformer object
    """
    if isinstance(model, FeatureModel):  # Check if we are dealing with a feature model
        return FeatureDataTransformer(dataset, model.params['data_params'], comparison_pw['text'])
    elif isinstance(model, PytorchModel):
        return PytorchDataTransformer(dataset, model.params['data_params'])
    elif isinstance(model, HuggingfaceModel):
        return PassGPT10Transformer(dataset, model.params['data_params'])
    else:
        return None


def run_test_for_model(model, params, test_file_name, comparison_pw, training=True, load_filename=None,
                       save_filename=None):
    metrics = {}

    dataset = load_from_disk(test_file_name)

    train_data = transform_data(model, dataset['train'], comparison_pw)
    test_data = transform_data(model, dataset['test'], comparison_pw)

    if load_filename is not None:
        model.load_model(load_filename)

    start_time = time.time()

    if training:
        model.train(train_data.X, train_data.y, params=params['train_params'])

    if save_filename is not None:
        model.save_model(save_filename)

    predictions = model.predict(test_data.X)

    accuracy = model.calc_accuracy(test_data.y, predictions)
    print(accuracy)
    metrics['accuracy'] = accuracy

    recall = model.calc_recall(test_data.y, predictions)
    print(recall)
    metrics['recall'] = recall

    precision = model.calc_precision(test_data.y, predictions)
    print(precision)
    metrics['precision'] = precision

    f1 = model.calc_f1score(test_data.y, predictions)
    print(f1)
    metrics['f1'] = f1

    metrics['running_time'] = time.time() - start_time

    '''
    second_model = PassGPT10Model(internet=False)
    second_model.load_model(save_filename)
    test_data = transform_data(second_model, dataset['test'], dataset['comparison_pw'], internet=internet)
    predictions = second_model.predict(test_data.X)
    accuracy = second_model.calc_accuracy(test_data.y, predictions)
    print(accuracy)
    '''

    '''
    dummy_dataset = create_dummy_dataset()
    dummy_data = transform_data(model, dummy_dataset, dataset['comparison_pw'], internet=internet)
    predictions = model.predict(dummy_data.X)
    print(predictions)
    accuracy = model.calc_accuracy(dummy_data.y, predictions)
    print(accuracy)
    '''

    return metrics


def print_dataset(dataset, split='train'):
    """
    Prints the whole dataset for a certain split of the dataset
    :param dataset: the dataset that need to be printed
    :param split: the split that needs to be printed (train, test, validation)
    :return: None
    """
    print(dataset[split]['text'])
    print(dataset[split]['label'])


def initialize_model(model_name, params):
    if model_name == "PassGPT10":
        return PassGPT10Model(params)
    if model_name == "DecisionTree":
        return DecisionTree(params)


def run_all_datasets(folder_name, model_name, params, comparison_pw, saving_folder_name):
    results = {}
    files = find_files(folder_name)
    for file in files:
        model = initialize_model(model_name, params)
        results[file] = run_test_for_model(model, params, folder_name + file, comparison_pw,
                                           save_filename=saving_folder_name + str(model) + "_" + file)
    with open(model_name, 'w') as f:
        json.dump(results, f, indent=4)
    return results


def create_all_models(params):
    return [RandomForest(params), LSTMModel(params), PassGPT10Model(params), GaussianNaiveBayes(params),
            MultinomialNaiveBayes(params), DecisionTree(params)]

def strip_filename(filename):
    return filename.split("/")[-1]

def param_grid_search(model_name, param_grid, params, test_file_name):
    model = initialize_model(model_name, params)

    dataset = load_from_disk(test_file_name)

    train_data = transform_data(model, dataset['train'], comparison_pw)

    clf = GridSearchCV(model.model, param_grid)
    clf.fit(train_data.X, train_data.y)

    print(clf.cv_results_)
    with open(model_name+"_FF_"+strip_filename(test_file_name), 'w') as f:
        json.dump(clf.cv_results_, f)

    print(clf.best_estimator_)
    print(clf.best_score_)


#print(find_files('datasets'))

comparison_pw = load_from_disk("comparison_pw")

# print(load_data_from_file('./testset_files/most_common_En1.0_1000'))
# test_filenames = ['']


# run_test_for_model(model, test_file_name)

# dataset = load_from_disk('./datasets/most_common_En1.0_100_split0')
# print(dataset)

# print_dataset(dataset)

"""
Three types of parameters:

model_params: parameters that determine properties of the model, such as the max_depth of a decision tree
data_params: parameters that determine how the data should be processed, such as whether n_gram features should be created or not
training_params: parameters that determine how training should proceed, such as the learning_rate or the nr of epochs

All these can be specified in a params dictionary, with subdictionaries for the aforementioned three types. 
The exact set of parameters recognized differs per model and should be specified in the model documentation. 
"""
internet = False

data_params = {}
model_params = {}
train_params = {}

data_params['levenshtein'] = False
data_params['ngrams'] = False
data_params['ngram_range'] = (1, 2)
data_params['internet'] = internet

train_params['epochs'] = 2

model_params['max_depth'] = 2
model_params['internet'] = internet

params = {'data_params': data_params,
          'train_params': train_params,
          'model_params': model_params}

# model = DecisionTree(params)
# model_name = "DecisionTree"

# model = LSTMModel()

# model = PassGPT10Model(params, load_filename="./models/PassGPT")
# model_name = "PassGPT10"
# model = initialize_model(model_name, params)
# run_test_for_model(model, params, './datasets/test/most_common_En1.0_1000_split0', comparison_pw,
#                   save_filename="./models/PassGPTmost_common_En1.0_1000_split0")
# run_test_for_model(model, params, './datasets/test/most_common_En1.0_1000_split2', comparison_pw,
#                   training=False, load_filename="./models/PassGPT")

# print(run_all_datasets("./datasets/test/", model_name, params, comparison_pw, "./models/"))

# print_dataset(load_from_disk('./datasets/test/most_common_En1.0_10000_split0'))
# print_dataset(load_from_disk('./datasets/test/most_common_En1.0_10000_split1'))


# ------------------------------- Parameter grid search -------------------------------

param_grid_DT = {'criterion': ('gini', 'entropy', 'log_loss'),
                 'splitter': ('best', 'random'),
                 'max_depth': [5, 10, 50, 100, 200, 500, None],
                 'min_samples_split': [2, 5, 10, 50, 100, 200, 500],
                 'min_samples_leaf': [1, 5, 10, 50, 100],
                 'class_weight': ('balanced', {0: 1, 1: 5}, {0: 1, 1: 10})}
model_name = "DecisionTree"
param_grid_search(model_name, param_grid_DT, params, './datasets/test/most_common_En1.0_1000_split0')
