import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import time
import json
from datasets import load_from_disk, Dataset
import pandas as pd

import sys

from sklearn.model_selection import GridSearchCV

from DataTransformer import FeatureDataTransformer, PytorchDataTransformer, PassGPT10Transformer, \
    ReformerDataTransformer
from FeatureModel import FeatureModel
from PytorchModel import PytorchModel, LSTMModel
from PassGPTModel import HuggingfaceModel, PassGPT10Model, ReformerModel
from FeatureModel import RandomForest, DecisionTree, GaussianNaiveBayes, MultinomialNaiveBayes, KNearestNeighbors, \
    AdaBoost

from utils import find_files_in_folder
from utils import grids


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


def transform_data(model, dataset, comparison_pw, split):
    """
    Transforms the data by using a DataTransformer that fits the model type
    :param model: the model the dataset is for
    :param dataset: the dataset with at least a train, test and validation part
    :param split: the part of the dataset you want; train, test or validation
    :return: the transformed data in a DataTransformer object
    """
    if isinstance(model, FeatureModel):  # Check if we are dealing with a feature model
        return FeatureDataTransformer(dataset, model.params['data_params'], comparison_pw['text'], model.vectorizer, split=split)
    elif isinstance(model, PytorchModel):
        return PytorchDataTransformer(dataset, model.params['data_params'])
    elif isinstance(model, PassGPT10Model):
        return PassGPT10Transformer(dataset, model.params['data_params'])
    elif isinstance(model, ReformerModel):
        return ReformerDataTransformer(dataset, model.params['data_params'])
    else:
        return None


def run_test_for_model(model, params, test_file_name, comparison_pw, training=True, load_filename=None,
                       save_filename=None, use_val=False, save_pred_folder=None):
    print(f"Running test on model {model} with file {test_file_name}")
    metrics = {}

    dataset = load_from_disk(test_file_name)

    if training:
        train_data = transform_data(model, dataset['train'], comparison_pw, 'training')
    if use_val:
        test_data = transform_data(model, dataset['validation'], comparison_pw, 'validation')
    else:
        test_data = transform_data(model, dataset['test'], comparison_pw, 'test')

    if load_filename is not None:
        model.load_model(load_filename)

    start_time = time.time()

    if training:
        model.train(train_data.X, train_data.y, params=params['train_params'])

    if save_filename is not None:
        model.save_model(save_filename)

    predictions = model.predict(test_data.X)

    if save_pred_folder is not None:
        # if nonexistent, build df with words, labels and predictions
        # if existent, load from csv and add column, save again
        test_file_last_part = test_file_name.split('/')[-1]
        files_in_save_pred_folder = find_files_in_folder(save_pred_folder)
        save_path = save_pred_folder+test_file_last_part + ".csv"
        if test_file_last_part + ".csv" not in files_in_save_pred_folder:
            if use_val:
                split = "validation"
            else:
                split = "test"
            dataset[split].to_pandas().to_csv(save_path)
        
        data_and_pred = pd.read_csv(save_path, index_col=0)
        data_and_pred[str(model) + "_" + test_file_last_part] = predictions
        data_and_pred.to_csv(save_path )
        

    accuracy = model.calc_accuracy(test_data.y, predictions)
    print(f"Accuracy: {accuracy}")
    metrics['accuracy'] = accuracy

    recall = model.calc_recall(test_data.y, predictions)
    print(f"Recall: {recall}")
    metrics['recall'] = recall

    precision = model.calc_precision(test_data.y, predictions)
    print(f"Precision: {precision}")
    metrics['precision'] = precision

    f1 = model.calc_f1score(test_data.y, predictions)
    print(f"F1: {f1}")
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
    if model_name == "PassGPT":
        return PassGPT10Model(params)
    if model_name == "Reformer":
        return ReformerModel(params)
    if model_name == "LSTM":
        return LSTMModel(params)
    if model_name == "DecisionTree":
        return DecisionTree(params)
    if model_name == "RandomForest":
        return RandomForest(params)
    if model_name == "NaiveBayes":
        return GaussianNaiveBayes(params)
    if model_name == "MultinomialNaiveBayes":
        return MultinomialNaiveBayes(params)
    if model_name == "KNearestNeighbors":
        return KNearestNeighbors(params)
    if model_name == "AdaBoost":
        return AdaBoost(params)


def run_all_datasets(folder_name, model_name, params, comparison_pw, saving_folder_name=None, use_val=False,
                     training=True, files=None, saved_models_folder=None, filter_part=None, save_pred_folder=None):
    results = {}
    if files is None:
        files = find_files_in_folder(folder_name)
    if filter_part is not None:
        files = filter_files(files, filter_part)
    for file in files:

        if saved_models_folder is not None:
            model_location = saved_models_folder + model_name + "Model" + "_" + file
            params['model_loc'] = model_location
            model = initialize_model(model_name, params)
            model.load_model(model_location)
        else:
            model = initialize_model(model_name, params)
        if saving_folder_name is None:
            save_filename = None
        else:
            save_filename = saving_folder_name + str(model) + "_" + file
        if model_name == "LSTM" and training:
            params['train_params']['fig_name'] = "Loss_LSTM_" + file + ".png"
        results[file] = run_test_for_model(model, params, folder_name + file, comparison_pw,
                                           save_filename=save_filename, use_val=use_val,
                                           training=training, save_pred_folder=save_pred_folder)
    print(results)
    with open(model_name, 'w') as f:
        json.dump(results, f, indent=4)
    return results


def create_all_models(params):
    return [RandomForest(params), LSTMModel(params), PassGPT10Model(params), GaussianNaiveBayes(params),
            MultinomialNaiveBayes(params), DecisionTree(params)]


def strip_filename(filename):
    return filename.split("/")[-1]


def param_grid_search(model_name, param_grid, params, test_file_name, save_folder="./gridsearchresults/"):
    model = initialize_model(model_name, params)

    dataset = load_from_disk(test_file_name)

    train_data = transform_data(model, dataset['train'], comparison_pw)

    start = time.time()
    clf = GridSearchCV(model.model, param_grid)
    clf.fit(train_data.X, train_data.y)
    duration = time.time() - start

    # save the grid search results to a csv file
    # code partially from https://medium.com/dvt-engineering/hyper-parameter-tuning-for-scikit-learn-ml-models-860747bc3d72
    results = pd.DataFrame(clf.cv_results_)
    results.to_csv(save_folder + model_name + "_" + strip_filename(test_file_name) + ".csv")

    results = results.loc[:, ('rank_test_score', 'mean_test_score', 'params')]
    results.sort_values(by='rank_test_score', ascending=True, inplace=True)

    return results, duration


def plot_decision_tree(model, test_file_name, params):
    dataset = load_from_disk(test_file_name)

    train_data = transform_data(model, dataset['train'], comparison_pw)

    model.train(train_data.X, train_data.y, params=params['train_params'])

    model.plot_tree("tree")


def filter_files(files, filter_part):
    return [file for file in files if filter_part not in file]


def extract_model_name_from_file_name(file_name):
    return file_name.split("_")[0][:-5]


def run_other_tests(models_folder_name, params, dataset_folder_name, comparison_pw, save_pred_folder=None):
    results = {}
    paths_to_models = find_files_in_folder(models_folder_name)
    model_types = [extract_model_name_from_file_name(filename) for filename in paths_to_models]
    datasets = find_files_in_folder(dataset_folder_name)

    for i, path in enumerate(paths_to_models):
        model_location = models_folder_name + path
        model = initialize_model(model_types[i], params)
        model.load_model(model_location)

        for dataset in datasets: 
            dataset_path = dataset_folder_name + dataset
            results[dataset + "+" + path] = run_test_for_model(model, params, dataset_path, comparison_pw,
                                           use_val=False, training=False, save_pred_folder=save_pred_folder)
    

    print(results)
    with open(model_name, 'w') as f:
        json.dump(results, f, indent=4)
    return results


#files = find_files_in_folder('datasets/def')
#print(files)
#print(filter_files(files, "50"))

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
internet = True

data_params = {}
model_params = {}
train_params = {}

data_params['levenshtein'] = False
data_params['ngrams'] = False
data_params['ngram_range'] = (1, 2)
data_params['internet'] = internet

train_params['epochs'] = 2
train_params['fig_name'] = 'test.png'  # For the LSTM

# optimal multinomial NB parameters
model_params['alpha'] = 1.0

# Decision Tree parameters
model_params['max_depth'] = None
model_params['weight'] = 'balanced'
model_params['min_samples_split'] = 10
model_params['min_samples_leaf'] = 1
model_params['criterion'] = 'gini'

# optimal RF parameters
model_params['n_estimators'] = 50 #watch out with same name parameter for AdaBoost
# rest of parameters is the same as DT

# optimal AdaBoost parameters
#model_params['n_estimators'] = 200 #comment out when using RF
model_params['learning_rate'] = 1.0

# optimal KNN parameters
model_params['n_neighbors'] = 50
model_params['metric'] = 'cosine'
model_params['weights'] = 'distance'

# --- OR ---
#model_params['metric'] = 'minkowski'
#model_params['p'] = 1

# LSTM parameters
model_params['batch_size'] = 64

# Huggingface model parameters
model_params['internet'] = internet

params = {'data_params': data_params,
          'train_params': train_params,
          'model_params': model_params}

# model = DecisionTree(params)
model_name = "DecisionTree"
#model_name = "KNearestNeighbors"
# model = LSTMModel()

# model = PassGPT10Model(params, load_filename="./models/PassGPT")
#model_name = "PassGPT"

#model_name = "AdaBoost"
#model_name = "DecisionTree"
#model_name = "LSTM"
# model_name = "MultinomialNaiveBayes"
#model = initialize_model(model_name, params)
#run_test_for_model(model, params, './datasets/def/most_common_En1.0_500000_split1', comparison_pw)
#print(model.model.get_depth())
#model.plot_tree("Tree_most_common_En1.0_500000_split1")
# run_test_for_model(model, params, './datasets/def/most_common_En1.0_1000_split0', comparison_pw,
#                   save_filename="./models/Reformer_most_common_En1.0_1000_split0")
# run_test_for_model(model, params, './datasets/def/most_common_En1.0_1000_split0', comparison_pw,
#                   training=False, load_filename="./models/Reformer_most_common_En1.0_1000_split0")

# run_test_for_model(model, params, './datasets/def/most_common_En1.0_10000_split0', comparison_pw,
#                   training=False, load_filename="./models/LSTMModel_most_common_En1.0_10000_split0")

#print(run_all_datasets("./datasets/def/", model_name, params, comparison_pw, "./models/",
#                       use_val=True, files=['most_common_En1.0_1000_split0']))
print(run_all_datasets("./datasets/def/", model_name, params, comparison_pw, saving_folder_name="./models/",
                       training=True, use_val=True, files=['most_common_En1.0_1000_split2']))
print(run_all_datasets("./datasets/def/", model_name, params, comparison_pw, saving_folder_name="./models/",
                       training=False, saved_models_folder="./models/", use_val=True, files=['most_common_En1.0_1000_split2'], save_pred_folder="./predictions/"))
#print(run_all_datasets("./datasets/def/", model_name, params, comparison_pw,
#                       training=True, use_val=True))

# print_dataset(load_from_disk('./datasets/def/most_common_En1.0_10000_split0'))
# print_dataset(load_from_disk('./datasets/def/most_common_En1.0_10000_split1'))

# datasetname=sys.argv[1]

#run_other_tests('./models/', params, './datasets/other_datasets/', comparison_pw, )

'''
run_test_for_model(model, params, f"./datasets/def/{datasetname}", comparison_pw,
                   save_filename=f"./models/PassGPT{datasetname}")
'''

# ------------------------------- Plotting decision tree -------------------------------
'''
model_name = "DecisionTree"
model_params['max_depth'] = 4
model = initialize_model(model_name, params)
plot_decision_tree(model, './datasets/most_common_En1.0_10000_split0', params)
'''

# ------------------------------- Parameter grid search -------------------------------


'''
dataset_files = find_files_in_folder('datasets/def')
model_name = "MultinomialNaiveBayes"
print(dataset_files)
for file in dataset_files:
    print(f"Running gridsearch for {file}")
    results, duration = param_grid_search(model_name, grids[model_name], params, './datasets/def/' + file)
    print(duration)
    print(results)


model_name = "RandomForest"
file = "most_common_En0.5Sp0.5_500000_split1"
results, duration = param_grid_search(model_name, grids[model_name], params, './datasets/def/' + file)
print(duration)
print(results)
'''

