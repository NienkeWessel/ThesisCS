import pandas as pd
from math import isclose
from utils import find_files_in_folder, split_column_title, read_in_csv
from datasets import load_from_disk
from DataTransformer import counts
import json

def calc_accuracy(df, label_column, pred_column):
    return (df[label_column] == df[pred_column]).mean()

def calc_recall(df, label_column, pred_column):
    TP = ((df[label_column] >= 0.95) & (df[pred_column] >= 0.95)).sum()
    FN = ((df[label_column] >= 0.95) & (df[pred_column] <= 0.05)).sum()

    # Calculate recall
    return TP / (TP + FN)

def calc_precision(df, label_column, pred_column):
    # Calculate true positives (TP) and false positives (FP)
    TP = ((df[label_column] >= 0.95) & (df[pred_column] >= 0.95)).sum()
    FP = ((df[label_column] <= 0.05) & (df[pred_column] >= 0.95)).sum()

    # Calculate precision
    return TP / (TP + FP)

def calc_f1score(df, label_column, pred_column):
    recall = calc_recall(df, label_column, pred_column)
    precision = calc_precision(df, label_column, pred_column)
    return 2 * (precision * recall) / (precision + recall)

def calc_weighted_fscore(df, label_column, pred_column, beta=2):
    recall = calc_recall(df, label_column, pred_column)
    precision = calc_precision(df, label_column, pred_column)
    return (1 + beta*beta) * (precision * recall) / ( (beta*beta *precision) + recall)




def calc_mean_length(df):
    return df['text'].apply(lambda x: len(x)).mean()

def try_is_lower(x):
    try: 
        return x.islower()
    except:
        return False

def try_is_numeric(x):
    try: 
        return x.isnumeric()
    except:
        return False


def percentage_of_lowercase_only(df):
    if len(df) > 0:
        return len(df[df['text'].apply(lambda x: try_is_lower(x))])/len(df)
    else:
        return 0

def percentage_of_numeric_only(df):
    if len(df) > 0:
        return len(df[df['text'].apply(lambda x: try_is_numeric(x))])/len(df)
    else:
        return 0

def run_counts(df):
    if len(df['text']) > 0:
        try: 
            return list(df['text'].apply(lambda x : counts(x, comparison_pw, levenshtein=False)).apply(pd.Series).mean())
        except:
            raise Exception(df['text'])
    else: 
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def apply_func_to_all_cats(function, df, column):
    stats = {
        #'total' : function(df),
        #'passwords': function(df[df['label'] >= 0.95]), 
        #'words' : function(df[df['label'] <= 0.05]),
        'TP' : function(df[(df['label'] >= 0.95) & (df[column] >= 0.95)]), 
        'FP' : function(df[(df['label'] <= 0.05) & (df[column] >= 0.95)]),
        'TN' : function(df[(df['label'] <= 0.05) & (df[column] <= 0.05)]),
        'FN' : function(df[(df['label'] >= 0.95) & (df[column] <= 0.05)]), 
    }
    return stats

def calc_stats_for_file(df, stats, file):
    models = df.columns[3:]
    models = [split_column_title(model) for model in models if model[-1] != '0' or model[-1] != '0testretry']

    functions = [run_counts, percentage_of_lowercase_only, percentage_of_numeric_only]
    function_names = ['counts', 'percentage lowercase only', 'percentage numeric only']

    for model in models:
        model_type = model[0] + "-" + model[3]
        if model_type not in stats:
            stats[model_type] = {}
        if model[1] == "":
            model_name = model[0] + "_" + file[:-4]
        else: 
            model_name = model[1]
        if model_name not in stats[model_type]:
            stats[model_type][model_name] = {}
        if file in stats[model_type][model_name]:
            print(f"Skipped {model_type} for {file}")
            continue
        stats[model_type][model_name][file] = {}
        for i, function in enumerate(functions):
            try: 
                stats[model_type][model_name][file][function_names[i]] = (apply_func_to_all_cats(function, df, "-".join(model)))
            except:
                 #print(stats)
                 raise Exception("problems here")
            
        stats[model_type][model_name][file]['accuracy'] = calc_accuracy(df, 'label', "-".join(model))
        stats[model_type][model_name][file]['f1score'] = calc_f1score(df, 'label', "-".join(model))
        stats[model_type][model_name][file]['recall'] = calc_recall(df, 'label', "-".join(model))
        stats[model_type][model_name][file]['precision'] = calc_precision(df, 'label', "-".join(model))


    return stats

def add_weighted_fscore_to_stats(df, stats, file):
    models = df.columns[3:]
    models = [split_column_title(model) for model in models if model[-1] != '0' or model[-1] != '0testretry']

    for model in models:
        
        model_type = model[0] + "-" + model[3]
        if model[1] == "":
            model_name = model[0] + "_" + file[:-4]
        else: 
            model_name = model[1]
        try: 
            stats[model_type][model_name][file]['weightedfscore5'] = calc_weighted_fscore(df, 'label', "-".join(model), beta=5)
        except: 
            raise Exception("problems here")

    return stats

def calc_stats_for_all_files(folder, existing_stats_file=None, method='normal'):
    files = find_files_in_folder(folder)
    if existing_stats_file is not None:
        with open(existing_stats_file, 'r') as f:
            stats = json.load(f)
    else: 
        stats = {}
    for file in files:
        print(f"Running file {file}")
        df = read_in_csv(folder + file)
        if method == 'normal':
            stats = calc_stats_for_file(df, stats, file)
        elif method == 'weighted':
            stats = add_weighted_fscore_to_stats(df, stats, file)
        else:
            print("Not a valid method, please choose either 'normal' or 'weighted'")
            return
        with open("Stats.json", 'w') as f:
            json.dump(stats, f, indent=4)
    #print(stats)


def merge_predictions(file_loc1, file_loc2, save_loc):
    """
    Merges the predictions from two csv files, and saves it into the save_loc
    :param file_loc1: the location of the file that needs to be first in the merged file
    :param file_loc2: the location of the file that needs to be second in the merged file
    :param save_loc: the location where the file needs to be saved
    :return: None
    """
    file1 = pd.read_csv(file_loc1, index_col=0)
    file2 = pd.read_csv(file_loc2, index_col=0)
    complete_df = pd.concat([file1, file2[file2.columns[2:]]], axis=1)
    complete_df.to_csv(save_loc)

def merge_all_prediction_files(folder1, folder2, saving_folder):
    """
    Merges all files in folder1 with files with matching filenames in folder2, and saves it into saving_folder
    :param folder1: the location of the folder with files that need to be first in the merged files
    :param folder2: the location of the folder with files that need to be second in the merged files
    :param saving_folder: the location where the files needs to be saved
    :return: None
    """
    files = find_files_in_folder(folder1)
    for file in files:
        merge_predictions(folder1 + file, folder2 + file, saving_folder + file)


def filter_lang_file(filename):
    file = pd.read_csv(filename, index_col=0)
    file = file[file['label'] == 0.0]
    file.to_csv(filename[:-4] + "_wordsonly.csv")

def move_PassGPT_results(stats_file="Stats.json"):
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    keys_to_be_deleted = []
    for model_type in stats:
        if model_type[-9:] == "testretry":
            print(model_type)
            stats[model_type[:-9]] = stats[model_type]
            keys_to_be_deleted.append(model_type)
    for k in keys_to_be_deleted:
        del stats[k]
    with open("Stats.json", 'w') as f:
        json.dump(stats, f, indent=4)

#merge_all_prediction_files("./predictionsSepFiles/predictionsFMs/", "./predictionsSepFiles/predictionsNNs/", "./predictions/")

comparison_pw = load_from_disk("comparison_pw")


#df = read_in_csv('./predictions/long_passwords_32.csv')
#print(calc_f1score(df, 'label', 'KNearestNeighborsModel-KNearestNeighborsMinkowskiModel_most_common_En0.5Sp0.5_1000_split2-long_passwords_16-Bigram'))

#print(calc_mean_length(df))

#print(apply_func_to_all_cats(calc_mean_length, df, 'KNearestNeighborsModel-KNearestNeighborsMinkowskiModel_most_common_En0.5Sp0.5_1000_split2-long_passwords_16-Bigram'))

#print(calc_stats_for_file(df))

#language_files = ['most_common_Ar1.0_10000.csv', 'most_common_Du1.0_10000.csv', 'most_common_It1.0_10000.csv', 'most_common_Ru1.0_10000.csv', 'most_common_Tu1.0_10000.csv', 'most_common_Vi1.0_10000.csv']
#for lang_file in language_files:
#    filter_lang_file(f"./predictionsSepFiles/predictionsBigrams/{lang_file}")

# ----- MAIN FUNCTION ------
calc_stats_for_all_files("./predictionsSepFiles/predictionsBigrams/", existing_stats_file="Stats.json", method='weighted')

#move_PassGPT_results()