import os
from datasets import load_from_disk, Dataset

from DataTransformer import FeatureDataTransformer, PytorchDataTransformer, PassGPT10Transformer
from FeatureModel import FeatureModel
from PytorchModel import PytorchModel
from PassGPTModel import HuggingfaceModel


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


def transform_data(model, dataset, comparison_pw, internet=False):
    """
    Transforms the data by using a DataTransformer that fits the model type
    :param model: the model the dataset is for
    :param dataset: the dataset with at least a train, test and validation part
    :param split: the part of the dataset you want; train, test or validation
    :param internet: whether you have internet access
    :return: the transformed data in a DataTransformer object
    """
    if isinstance(model, FeatureModel):  # Check if we are dealing with a feature model
        return FeatureDataTransformer(dataset, comparison_pw['text'])
    elif isinstance(model, PytorchModel):
        return PytorchDataTransformer(dataset)
    elif isinstance(model, HuggingfaceModel):
        return PassGPT10Transformer(dataset, internet=internet)
    else:
        return None


def run_test_for_model(model, test_file_name, comparison_pw, internet=False, save_filename=None):
    dataset = load_from_disk(test_file_name)

    train_data = transform_data(model, dataset['train'], comparison_pw, internet=internet)
    test_data = transform_data(model, dataset['test'], comparison_pw, internet=internet)

    model.train(train_data.X, train_data.y)

    if save_filename is not None:
        model.save_model(save_filename)

    predictions = model.predict(test_data.X)

    accuracy = model.calc_accuracy(test_data.y, predictions)
    print(accuracy)

    #recall = model.calc_recall(test_data.y, predictions)
    #print(recall)

    #precision = model.calc_precision(test_data.y, predictions)
    #print(precision)

    f1 = model.calc_f1score(test_data.y, predictions)
    print(f1)

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


def print_dataset(dataset, split='train'):
    """
    Prints the whole dataset for a certain split of the dataset
    :param dataset: the dataset that need to be printed
    :param split: the split that needs to be printed (train, test, validation)
    :return: None
    """
    print(dataset[split]['text'])
    print(dataset[split]['label'])


print(find_files('datasets'))

comparison_pw = load_from_disk("comparison_pw")

# print(load_data_from_file('./testset_files/most_common_En1.0_1000'))
# test_filenames = ['']


# run_test_for_model(model, test_file_name)

# dataset = load_from_disk('./datasets/most_common_En1.0_100_split0')
# print(dataset)

# print_dataset(dataset)


from FeatureModel import RandomForest
model = RandomForest()

from PytorchModel import LSTMModel
#model = LSTMModel()

internet = False
from PassGPTModel import PassGPT10Model
#model = PassGPT10Model(internet=internet)

run_test_for_model(model, './datasets/most_common_En1.0_1000_split2', comparison_pw, internet=internet, save_filename="blah")
