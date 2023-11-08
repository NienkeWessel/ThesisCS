# ThesisCS
Code for Master's thesis CS



## Data format
The repo has code to generate different datasets from certain files. If you want to add your own dataset, however, that is also possible. The code uses Huggingface datasets as its dataformat. It expects a 'text' and a 'label' column, with in the text the (pass)words and in the label column a 0.0 (not a password) or a 1.0 (password). If your data is already in that format, great! You should be able to use the code without any problems. If your data is not yet in that format, you need to transform it. For example, if you have a list of (pass)word strings in `word_list` and a list of labels in 0.0 or 1.0 format in `labels`, you can turn this into a dataset as follows:

```
dataset = Dataset.from_dict({
                    'text': word_list,
                    'label': labels
                })
```

If you want to use the complete code 'as is', you will need to specify a train, test and validation part in these datasets. If you have the splits in lists, you can use the `DatasetDict` from Huggingface to create a dictionary in a similar way: 

```
data = DatasetDict({
                'train': Dataset.from_dict({
                    'text': train_words,
                    'label': train_labels
                }),
                'validation': Dataset.from_dict({
                    'text': val_words,
                    'label': val_labels
                }),
                'test': Dataset.from_dict({
                    'text': test_words,
                    'label': test_labels
                }),
            })
```

## Parameters
Because of the many different models, there are also many different parameters to set. The parameters are divided up into three categories: 
- model_params: parameters that determine properties of the model, such as the max_depth of a decision tree
- data_params: parameters that determine how the data should be processed, such as whether n_gram features should be created or not
- training_params: parameters that determine how training should proceed, such as the learning_rate or the nr of epochs

What follows, if for each type of model, what parameters can be set. This gets complicated quite quickly but is sadly a consequence of employing a wide variety of models.

### Feature models


### Pytorch models

#### LSTM
- model_params: 
    * batch_size: the batch_size that the model handless

### Huggingface models

#### PassGPT10
- model_params:
    * internet: whether the computer on which everything is run has internet access

- data_params: 
    * internet: whether the computer on which everything is run has internet access
