import os
from MLModel import MLModel
#from readdata import load_data_from_file
from datasets import DatasetDict, load_from_disk

from DataTransformer import FeatureDataTransformer, PytorchDataTransformer
from FeatureModel import FeatureModel
from PytorchModel import PytorchModel

def find_files(dir):
    return os.listdir(dir)


def run_test_for_model(model, test_file_name):
    dataset = load_from_disk(test_file_name)

    if isinstance(model, FeatureModel): # Check if we are dealing with a feature model
        train_data = FeatureDataTransformer(dataset['train'], dataset['comparison_pw']['text'])
        test_data = FeatureDataTransformer(dataset['test'], dataset['comparison_pw']['text'])
    elif isinstance(model, PytorchModel):
        train_data = PytorchDataTransformer(dataset['train'])
        test_data = PytorchDataTransformer(dataset['test'])

    model.train(train_data.X, train_data.y)

    predictions = model.predict(test_data.X)
    accuracy = model.calc_accuracy(test_data.y, predictions)

    print(accuracy)


print(find_files('datasets'))

#print(load_data_from_file('./testset_files/most_common_En1.0_1000'))
#test_filenames = ['']


#run_test_for_model(model, test_file_name)

dataset = load_from_disk('./datasets/most_common_En1.0_1000')
print(dataset)



from FeatureModel import RandomForest
model = RandomForest()

#from PytorchModel import LSTMModel
#model = LSTMModel()
run_test_for_model(model, './datasets/most_common_En1.0_1000')