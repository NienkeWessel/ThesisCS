import json
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, load_from_disk, Dataset


def read_words_most_common(filename, nr_lines, words=None):
    if words is None:
        words = set()
    with open(filename, "r") as file:
        i = 0
        while i < nr_lines:
            word = next(file).strip().split("\t")[1]
            if (len(word) > 1 or word.isalnum()) and word not in words:
                words.add(word)
                i += 1
    return words


def read_passwords(filename, comp_nr_lines, nr_lines, comp_pw=None):
    passwords = []
    comparison_pw = []

    with open(filename, "r") as file:

        if comp_pw is None:
            for i in range(comp_nr_lines):
                comparison_pw.append(next(file).strip())
            i = 0
            while i < nr_lines:
                passwords.append(next(file).strip())
                i += 1
        else:
            i = 0
            comp = set(comp_pw['text'])
            while i < nr_lines:
                pw = next(file).strip()
                if pw not in comp:
                    passwords.append(pw)
                    i += 1

    return passwords, comparison_pw

def read_long_passwords(filename, min_length=16, skip=510000, save=True):
    passwords = []
    with open(filename, "r") as file:
        for i in range(0, skip):
            next(file)
        while True:
            try:
                pw = next(file).strip()
                if len(pw) >= min_length:
                    passwords.append(pw)
            except: 
                break
    
    pw_dataset = Dataset.from_dict({
        'text': passwords,
        'label': np.ones(len(passwords))
    })

    if save:
        data = DatasetDict({
                'test': pw_dataset,
            })
        data.save_to_disk(f"long_passwords_{min_length}")
    
    return pw_dataset

def make_languages_string(languages, lang_split):
    s = ""
    for language in languages:
        if language in lang_split:
            s += language[:2] + str(lang_split[language])
    return s


def create_comparison_pw_set(filename, comp_nr_lines=10000, save=False):
    _, comparison_pw = read_passwords(filename, comp_nr_lines, 0)
    comparison_pw = Dataset.from_dict({
        'text': comparison_pw,
        'label': np.ones(len(comparison_pw))
    })

    if save:
        comparison_pw.save_to_disk("comparison_pw")

    return comparison_pw


def read_passwords_and_words(pw_filename, languages, lang_files, lang_split, comp_nr_lines, nr_lines, comp_pw=None):
    passwords, _ = read_passwords(pw_filename, comp_nr_lines, nr_lines, comp_pw)
    words = set()
    for language in languages:
        if language in lang_split:
            nr_lines_per_lang = int(lang_split[language] * nr_lines)
            words = read_words_most_common(lang_files[language], nr_lines_per_lang, words)
    print("Nr of passwords: {}\nNr of words: {}".format(len(passwords), len(words)))

    all_words = passwords + list(words)
    labels = (np.concatenate((np.ones(len(passwords)), np.zeros(len(words))), axis=0)).tolist()
    return all_words, labels

def load_data(pw_filename, languages, lang_files, lang_split, comp_nr_lines, nr_lines, save=False, mode='most_common',
              train_split=0.7, test_val_ratio=0.5, nr_of_sets=3, comp_pw=None):
    
    all_words, labels = read_passwords_and_words(pw_filename, languages, lang_files, lang_split, comp_nr_lines, nr_lines, comp_pw=comp_pw)

    random_state = 42

    for n in range(nr_of_sets):

        train_words, test_val_words, train_labels, test_val_labels = train_test_split(all_words, labels,
                                                                                      train_size=train_split,
                                                                                      random_state=random_state + n)

        test_words, val_words, test_labels, val_labels = train_test_split(test_val_words, test_val_labels,
                                                                          train_size=test_val_ratio,
                                                                          random_state=random_state + n)

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
            })

            save_filename = mode + '_' + make_languages_string(languages, lang_split) + '_' + str(
                nr_lines) + "_split" + str(n)

            print(data)
            data.save_to_disk(save_filename)


def load_data_from_file(filename):
    with open(filename) as f:
        return json.load(f)


def create_password_list_from_file(source_filename, goal_filename, skip=510000, max=1000000):
    passwords = []
    with open(source_filename, "r") as file:
        for i in range(0, skip):
            next(file)
        while True:
            try:
                pw = next(file)
                passwords.append(pw)
            except: 
                break
    with open(goal_filename, 'w') as file:
        file.writelines(passwords)


def create_lang_testset(password_file, languages, lang_files, lang_split, nr_lines, mode="most_common", save=False, comp_pw=None):
    all_words, labels = read_passwords_and_words(password_file, languages, lang_files, lang_split, 0, nr_lines, comp_pw=comp_pw)

    dataset = Dataset.from_dict({
        'text': all_words,
        'label': labels
    })

    if save:
        save_filename = mode + '_' + make_languages_string(languages, lang_split) + '_' + str(nr_lines)
        data = DatasetDict({
                'test': dataset,
            })
        data.save_to_disk(save_filename)
    return

def read_usernames(username_file, nr_lines, save=False, comp_pw=None):
    usernames, _ = read_passwords(username_file, 0, nr_lines, comp_pw=None)

    dataset = Dataset.from_dict({
        'text': usernames,
        'label': np.zeros(len(usernames))
    })

    if save:
        save_filename = 'Usernames_' + str(nr_lines)
        data = DatasetDict({
                'test': dataset,
            })
        data.save_to_disk(save_filename)
    return

#pw_filename = "10-million-password-list-top-1000000.txt"
pw_filename = "xato-net-10-million-passwords.txt"
username_file = "xato-net-10-million-usernames.txt"
comp_nr_lines = 10000
#read_long_passwords(pw_filename, min_length=32)
#create_password_list_from_file(pw_filename, "random_other_pws.txt")


nr_lines_list = [50]#[1000, 10000, 100000, 500000]

# comparison_pw = create_comparison_pw_set(pw_filename, comp_nr_lines=comp_nr_lines, save=True)

comparison_pw = load_from_disk("comparison_pw")

languages = ['English', 'Spanish', 'Dutch', 'Arabic', 'Chinese', 'Russian', 'Turkish', 'Vietnamese', 'Italian']
lang_files = {
    'English': "eng_news_2020_1M-words.txt",
    'Spanish': "spa_news_2022_1M-words.txt",
    'Dutch': "nld_news_2022_1M-words.txt",
    'Arabic': "ara_news_2022_1M-words.txt",
    'Russian': "rus_news_2022_1M-words.txt",
    'Turkish': "tur_news_2022_1M-words.txt",
    'Vietnamese': "vie_news_2022_1M-words.txt",
    'Italian': "ita_news_2023_1M-words.txt"
}
lang_split = {
#    'English': 1.0,
#    'Spanish': 0.5,
#    'Dutch': 1.0,
    'Arabic': 1.0,
#    'Russian': 1.0,
#    'Turkish': 1.0,
#    'Vietnamese': 1.0,
#    'Italian' : 1.0,
}

#for nr_lines in nr_lines_list:
#    load_data(pw_filename, languages, lang_files, lang_split, 0, nr_lines, save=True, comp_pw=comparison_pw)

create_lang_testset("random_other_pws.txt", languages, lang_files, lang_split, 10000, save=True, comp_pw=comparison_pw)
#read_usernames(username_file, 10000, save=True, comp_pw=None)