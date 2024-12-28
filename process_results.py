import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("colorblind")
print(palette)
palette2 = sns.color_palette("Paired") #sns.cubehelix_palette(8) #sns.color_palette("magma", as_cmap=True)

sns.set_theme(font_scale=2.5)
import math
from os.path import exists

from utils import find_files_in_folder, split_column_title
from utils import grids, lang_map

from analyze_stats import read_stats_file, filter_filename_in_stats, transform_to_dataframe, filter_modeltype_in_stats

#from failure_analysis import calc_accuracy, calc_f1score, calc_precision, calc_recall

label_map = {
        'En1.0' : 'English or English/Spanish',
        'Du1.0' : 'Dutch',
        'It1.0' : 'Italian',
        'Tu1.0' : 'Turkish',
        'Vi1.0' : 'Vietnamese',
        'Ru1.0' : 'Russian',
        'Ar1.0' : 'Arabic',
        'En0.5Sp0.5' : 'English or English/Spanish',
        'long_passwords_24.csv' : '24+ chars',

}

label_map_pw = {
        'En1.0' : 'No min. length',
        'En0.5Sp0.5' : 'No min. length',
        'long_passwords_24.csv' : '24+ chars',
        'long_passwords_32.csv' : '32+ chars',
        'long_passwords_16.csv' : '16+ chars',
}

ordering = {
    'model' : ['NaiveBayes', 'MultinomialNaiveBayes', 'DecisionTree', 'RandomForest', 'AdaBoost', 'KNearestNeighborsCosine', 'KNearestNeighborsMinkowski']#['Base', 'Bigram', 'Levenshtein', 'LevenshteinBigram']
}

def find_base_filename(filename):
    return '_'.join(filename.split("_")[:-1])


def calc_averages(filename):
    """
    Calculates the averages from runs on multiple files. Assumes the last part of the original file name is the split
    number and can thus be discarded.

    :param filename: the filename where the results are in json format; for each filename one or multiple scores in
    a dictionary format
    :return: a dictionary with the averages per file type, discarding specific splits
    """
    with open(filename, 'r') as f:
        results = json.load(f)

    summarized_results = {}

    for full_filename in results:
        res_for_spec_file = results[full_filename]
        base_filename = find_base_filename(full_filename)
        if base_filename not in summarized_results:
            summarized_results[base_filename] = {
                'count': 1
            }
            for score_for_spec_file in res_for_spec_file:
                summarized_results[base_filename][score_for_spec_file] = [res_for_spec_file[score_for_spec_file]]
        else:
            summarized_results[base_filename]['count'] += 1
            for score_for_spec_file in res_for_spec_file:
                summarized_results[base_filename][score_for_spec_file].append(res_for_spec_file[score_for_spec_file])

    mean_results = {}

    for base_filename in summarized_results:
        mean_results[base_filename] = {}
        for score_for_all_files in summarized_results[base_filename]:
            if score_for_all_files != 'count':
                mean_results[base_filename][score_for_all_files] = np.mean(
                    summarized_results[base_filename][score_for_all_files])
                mean_results[base_filename][score_for_all_files + '_std'] = np.std(
                    summarized_results[base_filename][score_for_all_files])

    return mean_results


def transform_data(summary, model_name = "NaiveBayes"):
    table = pd.DataFrame.from_dict(summary, orient='index')
    table = table.rename_axis('dataset')
    table = table.reset_index()
    table['split'] = table.apply(lambda row: row['dataset'].split("_")[-1], axis=1)
    table['size'] = table.apply(lambda row: row['dataset'].split("_")[-2], axis=1)
    table['languages'] = table.apply(lambda row: lang_map[row['dataset'].split("_")[-3]], axis=1)
    table['model'] = model_name
    table.sort_values(by='size', ascending=True, inplace=True)
    return table


def plot_data(table, title, type_of_plot="languages", highlight=None):
    # Code inspired by https://engineeringfordatascience.com/posts/matplotlib_subplots/
    if type_of_plot == "modeltype" or type_of_plot == 'testing_set':
        scores = ['accuracy', 'recall', 'precision', 'f1score']
    else: 
        scores = ['accuracy', 'recall', 'precision', 'f1']
    fig, axs = plt.subplots(math.ceil(len(scores)/2), 2, figsize=(15,9), sharey=True)
    axs = axs.ravel()
    plt.suptitle(title, fontsize=30, y=0.98)
    print(table)
    for i, score in enumerate(scores):
        axs[i] = plt.subplot(math.ceil(len(scores)/2), 2, i+1)
        print(table[type_of_plot])
        axs[i] = sns.lineplot(data=table, x="size", y=score, ax=axs[i], hue=type_of_plot, style=type_of_plot, palette=palette) #, hue_order=ordering[type_of_plot])
        axs[i].get_legend().remove()
        axs[i].set(title=score)
        axs[i].set_ylabel("score")
        if highlight is not None:
            axs[i] = sns.lineplot(data=table[table["model"]== highlight], x="size", y=score, ax=axs[i], hue=type_of_plot, style=type_of_plot, color='black', linewidth=2.0)
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines,     # The line objects
           labels,   # The labels for each line
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title=type_of_plot,  # Title for the legend
           bbox_to_anchor = (0, -0.2, 1, 1),
           ncol=3,
           frameon=False
           )
    plt.tight_layout()
    plt.savefig(f'{title}.png', bbox_inches='tight')


def plot_languages_old_plots(table, title, languages=["100% English", "50% English, 50% Spanish"], highlight=None): # Unused
    label_map = {
        'En1.0' : '100% English or 50% English/50% Spanish',
        'Du1.0' : 'Dutch',
        'Ru1.0' : 'Russian',
        'It1.0' : 'Italian',
        'Vi1.0' : 'Vietnamese',
        'Ar1.0' : 'Arabic',
        'Tu1.0' : 'Turkish',
        'En0.5Sp0.5' : '100% English or 50% English/ 50% Spanish',
    }
    languages_in_label_map = list(label_map.keys())
    #print(len(languages)+ len(languages_in_label_map[:-1]))
    score = 'accuracy'
    fig, axs = plt.subplots(len(languages)+ len(languages_in_label_map[:-1]), 1, figsize=(15,15), sharey=True)
    axs = axs.ravel()
    plt.suptitle(title, fontsize=30, y=0.98)
    print(table)

    language_specific_dfs = []
    for lang in languages:
        language_specific_dfs.append(table[table["model_language"] == lang])

    for i, main_lang in enumerate(languages):
        axs[i] = plt.subplot(len(languages), 1, i+1)

        for j, lang in enumerate(languages):
            if lang == main_lang:
                alpha = 1.0
                axs[i] = sns.lineplot(data=language_specific_dfs[j], x="size", y=score, ax=axs[i], hue="testing_set", style="testing_set", palette=palette2, alpha=alpha, linewidth = 3) 
            else:
                alpha = 0.4
                axs[i] = sns.lineplot(data=language_specific_dfs[j], x="size", y=score, ax=axs[i], errorbar=None, hue="testing_set", style="testing_set", palette=palette2, alpha=alpha)

        
        axs[i].get_legend().remove()
        axs[i].set(title="Emphasis on " + main_lang + " models")
        axs[i].set_ylabel("Accuracy")
    

    testing_languages = languages_in_label_map[1:-1]
    language_specific_dfs = []
    for lang in testing_languages:
        language_specific_dfs.append(table[table["testing_set"] == lang])#.sort_values(by='testing_language', ascending=True, inplace=True, key=lambda s: s.apply(['En1.0', 'En0.5Sp0.5', 'Ar1.0', 'Du1.0', 'It1.0', 'Ru1.0', 'Tu1.0', 'Vi1.0'].index)))
        print(language_specific_dfs[-1])
    
    language_specific_dfs.append(table[(table["testing_set"] == 'En1.0') | (table["testing_set"] == 'En0.5Sp0.5')])
    print(len(language_specific_dfs))

    for i, test_lang in enumerate(language_specific_dfs):
        print(test_lang)
        axs[i+2] = sns.lineplot(data=test_lang, x="size", y=score, ax=axs[i+2], hue="model_language", style="model_language", palette=palette2, alpha=alpha, linewidth = 3) 


    lines, labels = fig.axes[0].get_legend_handles_labels()
    
    labels = [label_map[label] for label in labels][:int(len(labels)/2)]
    fig.legend(lines,     # The line objects
           labels,   # The labels for each line
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Testing language",  # Title for the legend
           bbox_to_anchor = (0, -0.1, 1, 1),
           ncol=4,
           frameon=False
           )
    plt.tight_layout()
    plt.savefig(f'{title}.png', bbox_inches='tight')


def plot_tiny_language_plots(table, title, save_folder, languages=["100% English", "50% English, 50% Spanish"], highlight=None):
    label_map_plot = list(label_map)[1:-1]

    score = 'adap_accuracy'
    fig, axs = plt.subplots(math.ceil(len(label_map_plot)/2), 2, figsize=(15,15), sharey=True)
    axs = axs.ravel()
    plt.suptitle(title, fontsize=30, y=0.98)

    for i, testing_lang in enumerate(label_map_plot):
        axs[i] = plt.subplot(math.ceil(len(label_map_plot)/2), 2, i+1)
        axs[i] = sns.lineplot(data=table[table["testing_set"] == testing_lang].sort_values(by=['size', 'model_language'], ascending=True, key=lambda s: 
                                                                                                s.apply(['1000', '10000', '100000', '500000', '100% English', '50% English, 50% Spanish'].index)), 
                                                                                                x="size", y=score, ax=axs[i], hue="model_language", style="model_language", palette=palette2, alpha=1.0, linewidth = 3) 
        axs[i].get_legend().remove()
        axs[i].set(title=label_map[testing_lang])
        axs[i].set_ylabel("Accuracy")
    

    lines, labels = fig.axes[0].get_legend_handles_labels()
    
    #labels = [label_map[label] for label in labels][:int(len(labels)/len(label_map_plot))]
    fig.legend(lines,     # The line objects
           labels,   # The labels for each line
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Model language",  # Title for the legend
           bbox_to_anchor = (0, -0.1, 1, 1),
           ncol=4,
           frameon=False
           )
    
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{title}.png', bbox_inches='tight')

''' Original specific one, trying to see if we can generalize it to also work for longer passwords
def plot_languages_two_plots(table, title, save_folder, languages=["100% English", "50% English, 50% Spanish"], highlight=None):
    
    score = 'adap_accuracy'
    fig, axs = plt.subplots(len(languages), 1, figsize=(15,15), sharey=True)
    axs = axs.ravel()
    plt.suptitle(title, fontsize=30, y=0.98)
    print(table)
    table.to_csv("languagefile_test.csv")

    language_specific_dfs = []
    for lang in languages:
        #print(lang)
        #print(language_specific_dfs)
        language_specific_dfs.append(table[table["model_language"] == lang].sort_values(by=['size', 'testing_language'], ascending=True, key=lambda s: s.apply(['1000', '10000', '100000', '500000', 'En1.0', 'En0.5Sp0.5', 'Ar1.0', 'Du1.0', 'It1.0', 'Ru1.0', 'Tu1.0', 'Vi1.0'].index)))
        #print("Printing hereeeee")
        #print(language_specific_dfs[-1])

    for i, main_lang in enumerate(languages):
        axs[i] = plt.subplot(len(languages), 1, i+1)

        for j, lang in enumerate(languages):
            if lang == main_lang:
                alpha = 1.0
                axs[i] = sns.lineplot(data=language_specific_dfs[j], x="size", y=score, ax=axs[i], hue="testing_language", style="testing_language", palette=palette2, alpha=alpha, linewidth = 3, errorbar=None) 
            #else:
            #    alpha = 0.4
            #    axs[i] = sns.lineplot(data=language_specific_dfs[j], x="size", y=score, ax=axs[i], errorbar=None, hue="testing_language", style="testing_language", palette=palette2, alpha=alpha)

        
        axs[i].get_legend().remove()
        axs[i].set(title=main_lang + " models")
        axs[i].set_ylabel("Accuracy")
    
    lines, labels = fig.axes[0].get_legend_handles_labels()
    labels = [label_map[label] for label in labels]
    fig.legend(lines,     # The line objects
           labels,   # The labels for each line
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Testing language",  # Title for the legend
           bbox_to_anchor = (0, -0.1, 1, 1),
           ncol=4,
           frameon=False
           )
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{title}.png', bbox_inches='tight')
'''

def plot_otherdatasets_two_plots(table, title, save_folder, languages=["100% English", "50% English, 50% Spanish"], plt_type='languages', highlight=None):
    settings = {
        'languages': {
            'score': 'adap_accuracy',
            'score_label' : 'Accuracy',
            'order' : ['En1.0', 'En0.5Sp0.5', 'Ar1.0', 'Du1.0', 'It1.0', 'Ru1.0', 'Tu1.0', 'Vi1.0'],
            'errorbar' : None,
            'label_map' : label_map
        },
        'long_pw' : {
            'score': 'recall',
            'score_label' : 'Recall',
            'order' : ['En1.0', 'En0.5Sp0.5', 'long_passwords_16.csv', 'long_passwords_24.csv', 'long_passwords_32.csv'],
            'errorbar' : ('ci', 95),
            'label_map' : label_map_pw
        }
    }
    score = settings[plt_type]['score']
    fig, axs = plt.subplots(len(languages), 1, figsize=(15,15), sharey=True)
    axs = axs.ravel()
    plt.suptitle(title, fontsize=30, y=0.98)
    print(table)
    #table.to_csv("languagefile_test.csv")

    language_specific_dfs = []
    for lang in languages:
        #print(lang)
        #print(language_specific_dfs)
        language_specific_dfs.append(table[table["model_language"] == lang].sort_values(by=['size', 'testing_set'], ascending=True, key=lambda s: s.apply((['1000', '10000', '100000', '500000'] + settings[plt_type]['order']).index)))
        #print("Printing hereeeee")
        #print(language_specific_dfs[-1])

    '''
    palette3 = []
    for a, b, c in palette2:
        palette3.append((a, b, max(c-0.3, 0.0)))
    '''

    for i, main_lang in enumerate(languages):
        axs[i] = plt.subplot(len(languages), 1, i+1)

        for j, lang in enumerate(languages):
            if lang == main_lang:
                alpha = 1.0
                axs[i] = sns.lineplot(data=language_specific_dfs[j], x="size", y=score, ax=axs[i], hue="testing_set", style="testing_set", palette=palette2, alpha=alpha, linewidth = 3, errorbar=settings[plt_type]['errorbar']) 
            #else:
            #    alpha = 0.4
            #    axs[i] = sns.lineplot(data=language_specific_dfs[j], x="size", y=score, ax=axs[i], errorbar=None, hue="testing_set", style="testing_set", palette=palette2, alpha=alpha)

        
        axs[i].get_legend().remove()
        axs[i].set(title=main_lang + " models")
        axs[i].set_ylabel(settings[plt_type]['score_label'])
    
    lines, labels = fig.axes[0].get_legend_handles_labels()
    labels = [settings[plt_type]['label_map'][label] for label in labels]
    fig.legend(lines,     # The line objects
           labels,   # The labels for each line
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Testing set",  # Title for the legend
           bbox_to_anchor = (0, -0.1, 1, 1),
           ncol=4,
           frameon=False
           )
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{title}.png', bbox_inches='tight')



def collect_all_files_and_filter(folder="ValSetResults/", lang=None):
    files = find_files_in_folder(folder)
    results = []
    for file in files:
        with open(folder + file, 'r') as f:
            results.append(transform_data(json.load(f), model_name=file))
    df = pd.concat(results)
    for mod in df['model'].unique():
        print(mod)
    if lang is not None:
        df = df[df["languages"] == lang]
    return df

def plot_diff_models(folder="ValSetResults/", lang=None, highlight=None):
    df = collect_all_files_and_filter(folder=folder, lang=lang)
    plot_data(df, "Scores of models", type_of_plot="model", highlight=highlight)


def plot_feature_effects(folder="ValSetResults/", lang=None):
    files = find_files_in_folder(folder)
    feature_models = ['AdaBoost', 'DecisionTree', 'RandomForest', 'KNearestNeighborsCosine', 'KNearestNeighborsMinkowski', 'MultinomialNaiveBayes', 'NaiveBayes']
    for feature_model in feature_models:
        print(feature_model)
        relevant_files = []
        for file in files:
            if feature_model in file and file[0] == feature_model[0]:
                relevant_files.append(file)
        results = []
        for file in relevant_files :
            with open(folder + file, 'r') as f:
                if len(feature_model) == len(file):
                    model_name = "Base"
                else: 
                    model_name = file[len(feature_model):]
                results.append(transform_data(json.load(f), model_name=model_name))
        df = pd.concat(results)
        title = f"Scores of {feature_model}"
        if lang is not None:
            df = df[df["languages"] == lang]
            title += f" ({lang})"
        plot_data(df, title, type_of_plot="model", highlight=None)

def collect_csvs(folder_name, model_name):
    files = find_files_in_folder(folder_name)
    filtered_files = []
    for file in files:
        if model_name in file:
            filtered_files.append(file)
    print(filtered_files)
    return filtered_files

def concat_all_csvs(folder_name, files, model_name):
    dataframes = []
    for file in files: 
        df = pd.read_csv(folder_name + file)
        df['filename'] = file
        dataframes.append(df)
    return pd.concat(dataframes)
            
def get_top_hyperparameters(folder_name, model_name, model_type="all", topn = 1):
    '''
    Prints how often settings appear in the top results for every hyperparameter
    :param folder_name: folder with the csv files with grid search results
    :param model_name: name of model that we are looking at; needs to be the same as the one in the csv filename and in the grid dict
    :param model_type: which dataset sizes do you want to consider. Options are: all, large (>50,000), small (<50,000), or an int, which is interpreted as a specific size
    :param topn: the top n in ranking you want to consider
    :return: None (prints the results)
    '''
    files = collect_csvs(folder_name, model_name)
    dataframes = concat_all_csvs(folder_name, files, model_name)
    dataframes['split'] = dataframes.apply(lambda row: (row['filename'].split("_")[-1]).split(".")[0], axis=1)
    dataframes['size'] = dataframes.apply(lambda row: int(row['filename'].split("_")[-2]), axis=1)
    dataframes['languages'] = dataframes.apply(lambda row: lang_map[row['filename'].split("_")[-3]], axis=1)

    top_scores = dataframes[dataframes['rank_test_score'] <= topn]

    if model_type == 'large':
        top_scores = top_scores[top_scores['size'] > 50000]
    if model_type == 'small':
        top_scores = top_scores[top_scores['size'] < 50000]
    if isinstance(model_type, int):
        top_scores = top_scores[top_scores['size'] == model_type]

    print(top_scores['params'].value_counts())

    for parameter in grids[model_name]:
        print(top_scores['param_' + parameter].value_counts())


def plot_differences(folder="ValSetResults/", lang1='100% English', lang2='50% English, 50% Spanish'):
    df = collect_all_files_and_filter(folder=folder, lang=None)
    scores = ['accuracy', 'recall', 'precision', 'f1']
    for score in scores:
        sizes = df['size'].unique()
        fig, axs = plt.subplots(len(sizes), 1, figsize=(15,25), sharey=True, sharex=True)
        axs = axs.ravel()
        plt.suptitle(f"Difference between 100% English and 50% English, 50% Spanish for {score}", fontsize=30, y=0.98)
        for i, size in enumerate(sizes):
            df_size = df[df['size'] == size]
            df_lang1 = df_size[df_size["languages"] == lang1]
            df_lang2 = df_size[df_size["languages"] == lang2]
            difference_languages = df_lang1.groupby(['model','split']).mean()-df_lang2.groupby(['model', 'split']).mean()
            difference_languages.reset_index(inplace=True)
            print(difference_languages)
            
            axs[i] = plt.subplot(len(sizes), 1, i+1)
            axs[i] = sns.barplot(difference_languages, x='model', y=score)
            #axs[i].get_legend().remove()
            axs[i].set(title=size)
            axs[i].set_ylabel(f"{score} difference")
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'LanguageDifferencesTest{score}.png', bbox_inches='tight')
    return



def calc_scores(df, column, ):
    return

def filter_feature_model_type(row, type):
    print(type)
    if type == 'base':
        return type in row['modeltype'] or '1' in row['modeltype'] or row['modeltype'][-1] == '-'
    elif type == 'Bigram': #we need to explicitly specify we don't want Levenshtein, so that it does not hit on BigramLevenshtein
        return type in row['modeltype'] and not 'Levenshtein' in row['modeltype']
    else: 
        return type in row['modeltype']

def filter_model_type(row, type):
    return type in row['modeltype']

def plot_all_languages_per_model(location):   # Unused? 
    # Read in all language files as dataframes
    # Get all headers from one of the dataframes

    language_files = ['most_common_Ar1.0_10000.csv', 'most_common_Du1.0_10000.csv', 'most_common_It1.0_10000.csv', 'most_common_Ru1.0_10000.csv', 'most_common_Tu1.0_10000.csv', 'most_common_Vi1.0_10000.csv']
    all_models = pd.read_csv(location + "most_common_En0.5Sp0.5_1000_split0.csv", index_col=0).columns[2:]
    for model in all_models.sort_values():
        print(model)
    '''
    # Split up the header into parts, and filter out the first column for the neural networks
    all_models = [split_column_title(model) for model in all_models if model[-1] != '0']
    print(all_models)

    model_types = set([model[1].split("_")[0] for model in all_models])
    print(model_types)

    
    for language_file in language_files:
        pd.read_csv(location + language_file, index_col=0)

    return
    '''

def plot_single_file_data(filename, title):
    stats = read_stats_file("Stats.json")
    filtered_stats = filter_filename_in_stats(stats, [filename])
    df = transform_to_dataframe(filtered_stats)
    df = df[df.apply(filter_feature_model_type, args=('BigramLevenshtein',), axis=1)]
    print(df)
    plot_data(df, title, type_of_plot="modeltype")

def generate_base_set_filenames():
    languages = ['En0.5Sp0.5', 'En1.0']
    sizes = ['1000', '10000', '100000', '500000']
    splits = ['0', '1', '2']
    filenames = []
    for lang in languages:
        for size in sizes:
            for split in splits: 
                filenames.append("most_common_" + lang + "_"+ size + "_split" + split + ".csv")
    return filenames

def extract_testing_lang(filename):
    if 'long_password' in filename:
        return filename
    else:
        return filename.split("_")[2]

def get_stats_df_from_files(files, modelname, folder, include_base=True):
    stats_filename = folder + modelname + "_stats.csv"
    if not exists(stats_filename):
        stats = read_stats_file("Stats.json")
        filtered_stats = filter_modeltype_in_stats(stats, modelname)
        if include_base:
            base_filenames = generate_base_set_filenames()
            filtered_stats = filter_filename_in_stats(filtered_stats, files + base_filenames)
        else:
            filtered_stats = filter_filename_in_stats(filtered_stats, files)
        print(filtered_stats)
        df = transform_to_dataframe(filtered_stats)
        df.to_csv(stats_filename)
    else:
        df = pd.read_csv(stats_filename, index_col=0)
    return df

def generate_questionnaire_stats(folder,testtype="questionnaire_data.csv"):
    stats_file_loc = folder+testtype[:-4] + "stats.csv"
    if not exists(stats_file_loc):
        stats = read_stats_file("Stats.json")
        questionnaire_stats = filter_filename_in_stats(stats, [testtype])
        df = transform_to_dataframe(questionnaire_stats)
        df.to_csv(stats_file_loc)
    else: 
        df = pd.read_csv(stats_file_loc, index_col=0)
    return df

def recompute_accuracy(row, folder='predictions/'):
    if row['testing_set'] == 'En1.0' or row['testing_set'] == 'En0.5Sp0.5':
        #print(row)
        df = pd.read_csv(folder + row['filename'])
        nr_of_pw = df['label'].sum()
        return (float(len(df)) * row['accuracy'] - nr_of_pw * row['recall']) / (float(len(df))-nr_of_pw)
    else:
        return row['accuracy']


def plot_languagesdata_for_model(modelname, title, model_type, folder="./langplotfiles/", save_folder="./languageGraphs"):
    language_files = ['most_common_Ar1.0_10000_wordsonly.csv', 'most_common_Du1.0_10000_wordsonly.csv', 'most_common_It1.0_10000_wordsonly.csv', 'most_common_Ru1.0_10000_wordsonly.csv', 'most_common_Tu1.0_10000_wordsonly.csv', 'most_common_Vi1.0_10000_wordsonly.csv']
    df = get_stats_df_from_files(language_files, modelname, folder)
    df['size'] = df.apply(lambda row: str(row['size']), axis=1)
    print(df)
    df.to_csv("languagefile_test2.csv") # for debugging purposes
    if model_type != "NN":
        df = df[df.apply(filter_feature_model_type, args=(model_type,), axis=1)]
    print(df)
    df['testing_set'] = df['filename'].apply(extract_testing_lang)
    df['adap_accuracy'] = df.apply(recompute_accuracy, axis=1)
    df.sort_values(by='size', ascending=True, inplace=True)
    #df = df[df.apply(filter_model_type, args=(modelname,), axis=1)]
    print(df)
    plot_otherdatasets_two_plots(df, title, save_folder)
    plot_tiny_language_plots(df, title+" (per language)", save_folder)


def plot_longpasswords(modelname, title, model_type, folder="./longpasswordfiles/", save_folder="./passwordGraphs"):
    password_files = ['long_passwords_16.csv', 'long_passwords_24.csv', 'long_passwords_32.csv']
    df = get_stats_df_from_files(password_files, modelname, folder)
    df['size'] = df.apply(lambda row: str(row['size']), axis=1)
    print(df)
    #df.to_csv("languagefile_test2.csv") # for debugging purposes
    if model_type != "NN":
        df = df[df.apply(filter_feature_model_type, args=(model_type,), axis=1)]
    print(df)
    df['testing_set'] = df['filename'].apply(extract_testing_lang)
    #df['adap_accuracy'] = df.apply(recompute_accuracy, axis=1)
    df.sort_values(by='size', ascending=True, inplace=True)
    #df = df[df.apply(filter_model_type, args=(modelname,), axis=1)]
    print(df)
    plot_otherdatasets_two_plots(df, title, save_folder, plt_type='long_pw')


def filter_models_for_barplot(row, filters):
    return row['modeltype'] not in filters

def plot_questionnaire(file_to_plot= 'questionnaire_data.csv', folder="./questionnairefiles/", save_folder="./questionnaireGraphs"):
    df = generate_questionnaire_stats(folder, testtype=file_to_plot)
    df = df[df.apply(filter_models_for_barplot, args=(['PassGPTModel-1testretry', 'PassGPTModel-0testretry', 'PassGPTModel-0', 'LSTMModel-0',
                                                       'KNearestNeighborsModel-Bigram', 'KNearestNeighborsModel-BigramLevenshtein', 'KNearestNeighborsModel-Levenshtein', 'KNearestNeighborsminkowksiModel-Bigram'], ), axis=1)]
    df['size'] = df.apply(lambda row: str(row['size']), axis=1)
    print(df)
    df.sort_values(by=['size', 'modeltype'], ascending=True, inplace=True)

    scores = ['accuracy', 'recall', 'precision', 'f1score']
    for score in scores:
        sizes = df['size'].unique()
        fig, axs = plt.subplots(len(sizes), 1, figsize=(15,25), sharey=True, sharex=True)
        axs = axs.ravel()
        plt.suptitle(f"Questionnaire data ({score})", fontsize=30, y=0.98)
        for i, size in enumerate(sizes):
            df_size = df[df['size'] == size]
            
            axs[i] = plt.subplot(len(sizes), 1, i+1)
            axs[i] = sns.barplot(df_size, x='modeltype', y=score)
            axs[i].set(title=size)
            axs[i].set(ylim=(0.0, 1.0))
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'{save_folder}/{file_to_plot[:-4]}-{score}.png', bbox_inches='tight')

    





#get_top_hyperparameters("./gridsearchresults/", "RandomForest", model_type='all', topn=1)

'''
#summary = calc_averages('ValSetResults/NaiveBayes')
with open('ValSetResults/NaiveBayes', 'r') as f:
    results = json.load(f)
table = transform_data(results)
plot_data(table, "Gaussian Naive Bayes")
'''

plot_diff_models(folder="ValSetResults/basemodels/", lang="100% English")
#plot_feature_effects(lang="100% English")

#plot_differences(folder="ValSetResults/all/")

#plot_all_languages_per_model("./predictions/")

#plot_single_file_data("questionnaire_data.csv", "Questionnaire data")

#print(generate_base_set_filenames())

models = {
    'LSTM' : {
        'file' : 'LSTMModel-1',
        'title' : 'LSTM',
        'type' : 'NN', 
    },
    'PassGPT' : {
        'file' : 'PassGPTModel-1',
        'title' : 'PassGPT',
        'type' : 'NN',
    }
}
feature_models = ['AdaBoost', 'RandomForest', 'DecisionTree',
                  'MultinomialNB', 'GaussianNB']

for feature_model in feature_models:
    model_types = ['base', 'Bigram', 'BigramLevenshtein', 'Levenshtein']
    for typ in model_types: 
        if typ == "BigramLevenshtein":
            title = "bigram + Levenshtein"
        elif typ == "Bigram":
            title = "bigram"
        else: 
            title = typ
        models[feature_model + "-" + typ] = {
            'file' : feature_model + "Model-" + typ,
            'title' : feature_model + f" ({title})",
            'type' : typ
        }


model = 'AdaBoost-Bigram'
#plot_languagesdata_for_model(models[model]['file'], f"{models[model]['title']} on different languages", models[model]['type'])
#plot_questionnaire(file_to_plot= 'Usernames_10000.csv', folder="./questionnairefiles/", save_folder="./questionnaireGraphs")
#for model in models:
#    plot_longpasswords(models[model]['file'], f"{models[model]['title']} on different languages", models[model]['type'])

#stats = read_stats_file("Stats.json")
#print(stats['KNearestNeighborscosineModel-'])

#print(list(models.keys())[14:])

#for model in models:#list(models.keys())[14:]:
#    plot_languagesdata_for_model(models[model]['file'], f"{models[model]['title']} on different languages", models[model]['type'])



