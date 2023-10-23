import json
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, load_from_disk, Dataset


def read_words(filename, nr_lines, words=None):
    if words is None:
        words = []
    with open(filename, "r") as file:
        for i in range(nr_lines):
            words.append(next(file).strip().split("\t")[1])
    return words


def read_passwords(filename, comp_nr_lines, nr_lines):
    passwords = []
    comparison_pw = []

    with open(filename, "r") as file:
        for i in range(comp_nr_lines):
            comparison_pw.append(next(file).strip())
        for i in range(nr_lines):
            passwords.append(next(file).strip())

    return passwords, comparison_pw


def make_languages_string(languages, lang_split):
    s = ""
    for language in languages:
        if language in lang_split:
            s += language[:2] + str(lang_split[language])
    return s


def load_data(pw_filename, languages, lang_files, lang_split, comp_nr_lines, nr_lines, save=False, mode='most_common',
              train_split=0.7, test_val_ratio=0.5, nr_of_sets=3):
    passwords, comparison_pw = read_passwords(pw_filename, comp_nr_lines, nr_lines)
    words = []
    for language in languages:
        if language in lang_split:
            nr_lines_per_lang = int(lang_split[language] * nr_lines)
            words = read_words(lang_files[language], nr_lines_per_lang, words)
    # print(words)
    print("Nr of passwords: {}\nNr of words: {}".format(len(passwords), len(words)))

    all_words = passwords + words
    labels = (np.concatenate((np.ones(len(passwords)), np.zeros(len(words))), axis=0)).tolist()

    random_state = 42

    for n in range(nr_of_sets):

        train_words, test_val_words, train_labels, test_val_labels = train_test_split(all_words, labels,
                                                                                      train_size=train_split,
                                                                                      random_state=random_state)

        test_words, val_words, test_labels, val_labels = train_test_split(test_val_words, test_val_labels,
                                                                          train_size=test_val_ratio,
                                                                          random_state=random_state)

        print("Training set size: {}\nTest set size: {}\nValidation set size: {}".format(len(train_words),
                                                                                         len(test_words),
                                                                                         len(val_words)))

        if save:
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
                'comparison_pw': Dataset.from_dict({
                    'text': comparison_pw,
                    'label': np.ones(len(comparison_pw))
                }),
            })
            '''
           'text': passwords + words,
           'labels': (np.concatenate((np.ones(len(passwords)), np.zeros(len(words))), axis=0)).tolist(),
           'comp_pw': comparison_pw,
           '''

            save_filename = mode + '_' + make_languages_string(languages, lang_split) + '_' + str(
                nr_lines) + "_split" + str(n)
            # with open(save_filename, 'w') as f:
            #    json.dump(data, f)
            print(data)
            data.save_to_disk(save_filename)
    return passwords, words, comparison_pw


def load_data_from_file(filename):
    with open(filename) as f:
        return json.load(f)



languages = ['English', 'Spanish', 'Dutch', 'Arabic', 'Chinese']
lang_files = {
    'English': "eng_news_2020_1M-words.txt",
    'Spanish': "eng_news_2020_1M-words.txt"
}
lang_split = {
    'English': 1.0,
    # 'Spanish': 0.0
}
pw_filename = "10-million-password-list-top-1000000.txt"
comp_nr_lines = 10000
nr_lines = 1000

load_data(pw_filename, languages, lang_files, lang_split, comp_nr_lines, nr_lines, save=True)

s = make_languages_string(languages, lang_split)
print(s)

