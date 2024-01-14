import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("colorblind")
sns.set(font_scale=1.5)
import math

lang_map = {
    'En0.5Sp0.5' : "50% English, 50% Spanish",
    'En1.0' : "100% English"
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


def transform_data(summary):
    table = pd.DataFrame.from_dict(summary, orient='index')
    table = table.rename_axis('dataset')
    table = table.reset_index()
    table['split'] = table.apply(lambda row: row['dataset'].split("_")[-1], axis=1)
    table['size'] = table.apply(lambda row: row['dataset'].split("_")[-2], axis=1)
    table['languages'] = table.apply(lambda row: lang_map[row['dataset'].split("_")[-3]], axis=1)
    table.sort_values(by='size', ascending=True, inplace=True)
    return table


def plot_data(table, title):
    # Code inspired by https://engineeringfordatascience.com/posts/matplotlib_subplots/
    scores = ['accuracy', 'recall', 'precision', 'f1']
    fig, axs = plt.subplots(math.ceil(len(scores)/2), 2, figsize=(15,9), sharey=True)
    axs = axs.ravel()
    plt.suptitle(title, fontsize=30, y=0.98)
    print(table)
    for i, score in enumerate(scores):
        axs[i] = plt.subplot(math.ceil(len(scores)/2), 2, i+1)
        axs[i] = sns.lineplot(data=table, x="size", y=score, ax=axs[i], hue="languages", style="languages", palette=palette)
        axs[i].get_legend().remove()
        axs[i].set(title=score)
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines,     # The line objects
           labels,   # The labels for each line
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Dataset languages",  # Title for the legend
           bbox_to_anchor = (0, -0.1, 1, 1),
           ncol=4,
           frameon=False
           )
    plt.tight_layout()
    plt.savefig(f'{title}.png', bbox_inches='tight')


#summary = calc_averages('ValSetResults/NaiveBayes')
with open('ValSetResults/NaiveBayes', 'r') as f:
    results = json.load(f)
table = transform_data(results)
plot_data(table, "Gaussian Naive Bayes")
