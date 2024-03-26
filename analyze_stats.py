import json
import pandas as pd
from utils import lang_map

def read_stats_file(filename):
    with open(filename, 'r') as f:
        stats = json.load(f)
    return stats

def filter_filename_in_stats(stats, filename):
    new_stats = {}
    for modeltype in stats:
        for modeln in stats[modeltype]:
            for filen in stats[modeltype][modeln]:
                if filen == filename:
                    if modeltype not in new_stats:
                        new_stats[modeltype] = {}
                    if modeln not in new_stats[modeltype]:
                        new_stats[modeltype][modeln] = {}
                    new_stats[modeltype][modeln][filename] = stats[modeltype][modeln][filename]
    return new_stats

def filter_modelname_in_stats(stats, modelname):
    new_stats = {}
    for modeltype in stats:
        for modeln in stats[modeltype]:
            if modeln == modelname:
                if modeltype not in new_stats:
                    new_stats[modeltype] = {}
                new_stats[modeltype][modelname] = stats[modeltype][modelname]
    return new_stats

def filter_modeltype_from_stats(stats, modeltype):
    return stats[modeltype]

def transform_to_dataframe(stats):
    scores = ['accuracy', 'f1score', 'recall', 'precision']
    parts = ['TP', 'FP', 'TN', 'FN']
    stats_names = ['percentage lowercase only', 'percentage numeric only']

    combination = []
    for part in parts:
        for stat in stats_names:
            combination.append(part + "-" + stat)
    
    # length, alpha_lower, alpha_lower / length, alpha_upper, alpha_upper / length, numeric,
    # numeric / length, special, special / length, char_sets, count_non_repeating(s)
    count_parts = ['length', 'alpha_lower', 'ratio_alpha_lower', 'alpha_upper', 'ratio_alpha_upper', 'numeric', 'ratio_numeric', 'special', 'ratio_special', 'char_sets', 'non_repeating_length']

    combination_counts = []
    for part in parts:
        for count_part in count_parts:
            combination_counts.append(part + "-" + count_part)

    df = pd.DataFrame(columns=['modeltype', 'modelname', 'filename']+ scores + combination + combination_counts)

    for model_type in stats:
        print(model_type)
        for model_name in stats[model_type]:
            #print(model_name)
            for file in stats[model_type][model_name]:
                #print(file)
                datarow = [model_type, model_name, file]

                # read out scores
                for score in scores:
                    datarow.append(stats[model_type][model_name][file][score])

                # read out the extra stats per part
                for part in parts:
                    for stat in stats_names:
                        datarow.append(stats[model_type][model_name][file][stat][part])

                # read out the counts
                for part in parts:
                    for i, _ in enumerate(count_parts):
                        datarow.append(stats[model_type][model_name][file]['counts'][part][i])

                # put it all in a dataframe row and append
                #print(datarow)
                df = df.append(pd.Series(datarow, index=df.columns), ignore_index=True)
    
    df['split'] = df.apply(lambda row: row['modelname'].split("_")[-1], axis=1)
    df['size'] = df.apply(lambda row: row['modelname'].split("_")[-2], axis=1)
    df['model_language'] = df.apply(lambda row: lang_map[row['modelname'].split("_")[-3]], axis=1)
    return df

'''
stats = read_stats_file("Stats.json")
filtered_stats = filter_filename_in_stats(stats, "questionnaire_data.csv")
print(transform_to_dataframe(filtered_stats))
'''