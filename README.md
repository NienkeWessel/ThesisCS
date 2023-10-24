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
