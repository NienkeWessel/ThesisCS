import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def transform_data(summary):
    table = pd.DataFrame.from_dict(summary, orient='index')
    table = table.rename_axis('dataset')
    table = table.reset_index()
    table.sort_values(by='dataset', ascending=True, inplace=True)
    table['size'] = table.apply(lambda row: row['dataset'].split("_")[-1], axis=1)
    table['languages'] = table.apply(lambda row: row['dataset'].split("_")[-2], axis=1)
    return table


def plot_data(table):
    scores = ['accuracy', 'recall', 'precision', 'f1']
    table = table[table['languages'] == 'En1.0']
    fig, ax = plt.subplots()
    print(table)
    for score in scores:
        ax = sns.lineplot(data=table, x="size", y=score, errorbar=None, label=score, ax=ax)
        ax.fill_between(table['size'], y1=table[score] - table[score+"_std"], y2=table[score] + table[score+"_std"], alpha=0.2)
    plt.legend()
    ax.set_ylabel("score")
    plt.savefig('test.png')


summary = calc_averages('ValSetResults/NaiveBayes')
table = transform_data(summary)
plot_data(table)
