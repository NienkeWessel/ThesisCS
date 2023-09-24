def read_words(filename, nr_lines, words=[]):
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


def load_data(pw_filename, languages, lang_files, lang_split, comp_nr_lines, nr_lines):
    passwords, comparison_pw = read_passwords(pw_filename, comp_nr_lines, nr_lines)
    words = []
    for language in languages:
        if language in lang_split:
            nr_lines_per_lang = int(lang_split[language] * nr_lines)
            words = read_words(lang_files[language], nr_lines_per_lang, words)
    # print(words)
    print("Nr of passwords: {}\nNr of words: {}".format(len(passwords), len(words)))

    return passwords, words, comparison_pw
