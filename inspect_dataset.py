from datasets import load_from_disk

def print_dataset(dataset, split='train'):
    """
    Prints the whole dataset for a certain split of the dataset
    :param dataset: the dataset that need to be printed
    :param split: the split that needs to be printed (train, test, validation)
    :return: None
    """
    print(dataset[split]['text'])
    print(dataset[split]['label'])

def filter_data(dataset, split='test', min_length=16):
    #return dataset
    split = dataset[split]
    return split.filter(lambda x: x['label'] == 1.0 and len(x['text'])>=min_length)

print_dataset(load_from_disk('./most_common_Ar1.0_1000_split0'))
print(load_from_disk('./long_passwords')['text'][:30])
#print(filter_data(load_from_disk('./most_common_Ar1.0_500000_split0'))['text'])