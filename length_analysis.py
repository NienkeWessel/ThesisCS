import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calc_corrected_answers(row, colname):
    if row['label'] == 0:
        return 1 - row[colname]
    else:
        return row[colname]

def analyze_file_for_length(filename, modelname, xmax=20, plottitle=None):
    df = pd.read_csv(filename, index_col=0)
    df['length'] = df['text'].apply(lambda x: len(str(x)))
    print(df)

    # Filter out lengths higher than xmax to keep plot feasible (otherwise x-axis is to large)
    df = df[df['length'] <= xmax]

    df['model_cor'] = df.apply(calc_corrected_answers, args=(modelname, ), axis=1)

    g = sns.barplot(df, x='length', y='model_cor',  hue="label", alpha=0.6)

    #g.set_axis_labels("Length", "Accuracy")
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels = ['Words', 'Passwords'], loc='lower left')

    if plottitle is not None:
        title = plottitle
    else:
        title = modelname

    g.set(xlabel ="Length", ylabel = "Accuracy", title=title)

    file = filename.split("/")[-1]
    plt.savefig(f'failureanalysis/{file[:-4]}-{modelname}.png', bbox_inches='tight')


def combine_cols(row):
    if row['lowercase'] and row['passwords']:
        return "Lowercase passwords"
    elif row['lowercase'] and not row['passwords']: 
        return "Lowercase words"
    elif not row['lowercase'] and row['passwords']: 
        return "Other passwords"
    elif not row['lowercase'] and not row['passwords']: 
        return "Other words"
    else: 
        return "Other"


def remove_split_part(st):
    if "_" not in st: #one of the first two columns which are not a model
        return st
    elif st[-1] == "0" or st[-1] == "1" or st[-1] == "-":
        return ("_").join(st.split("_")[:-1] ) if "_" in st else st
    else: # Cases with bigrams and such; don't even try to understand what is happening here :'(
        return st.split("-")[0] + "-" + st.split("-")[-1] + "-" + ("_").join(st.split("-")[-2].split("_")[:-1]) 

def analyze_file_for_lowercase(filename, plottitle=None): 

    df = pd.DataFrame()
    for split in [0, 1, 2]:
        df1 = pd.read_csv(filename + f"_split{split}.csv", index_col=0)
        print(df1.columns)
        df1 = df1.drop([x for x in df1.columns if x[-2:] == "-0"], axis=1)
        df1.columns = df1.columns.map(lambda x: remove_split_part(x) )
        df = pd.concat([df, df1], ignore_index=True)
        print(df)

    df['islowercaseonly'] = df['text'].apply(lambda x: str(x).islower())
    #print(df)

    #print(df.columns[2:-1])

    onlylowercasedf_pw = df[df['islowercaseonly']&df['label'] == 1.0]
    otherdf_pw = df[~df['islowercaseonly']&df['label'] == 1.0]
    onlylowercasedf_word = df[df['islowercaseonly']&df['label'] == 0.0]
    otherdf_word = df[~df['islowercaseonly']&df['label'] == 0.0]
    print(onlylowercasedf_pw)

    newcols = ('modeltype', 'mean', 'std', 'lowercase', 'passwords')
    new_df = pd.DataFrame(columns=newcols)

    rows = []
    print(df.columns[2:-1])
    for column in df.columns[2:-1]:
        rows.append({
            'modeltype': column, 
            'mean' : onlylowercasedf_pw[column].mean(), 
            'std' : onlylowercasedf_pw[column].std(), 
            'lowercase' : True,
            'passwords' : True
            })
        rows.append({
            'modeltype': column, 
            'mean' : 1-onlylowercasedf_word[column].mean(), 
            'std' : onlylowercasedf_word[column].std(), 
            'lowercase' : True,
            'passwords' : False
            })
        rows.append({
            'modeltype': column, 
            'mean' : otherdf_pw[column].mean(), 
            'std' : otherdf_pw[column].std(), 
            'lowercase' : False,
            'passwords' : True
            })
        rows.append({
            'modeltype': column, 
            'mean' : 1-otherdf_word[column].mean(), 
            'std' : otherdf_word[column].std(), 
            'lowercase' : False,
            'passwords' : False
            })
    print(pd.DataFrame(rows, columns=newcols))
    new_df = pd.concat([new_df, pd.DataFrame(rows, columns=newcols)], ignore_index=True)

    new_df['modeltype'] = new_df['modeltype'].apply(lambda x: "".join(x.split("-")[:2]).replace("Model", ""))

    new_df['type'] = new_df.apply(combine_cols, axis=1)
    #print(new_df)
    new_df.sort_values(by=['modeltype'], ascending=True, inplace=True)
    new_df = new_df[new_df['modeltype'] != "KNearestNeighbors"]


    sns.barplot(new_df, x='modeltype', y='mean',  hue='type')
    file = filename.split("/")[-1]
    plt.legend(bbox_to_anchor=(0.5, 1.2), loc="upper center", ncol=2)
    plt.xticks(rotation=90)
    plt.savefig(f'failureanalysis/lowercase_{file}.png', bbox_inches='tight')
    '''
    df[modelname + '_model_cor'] = df.apply(calc_corrected_answers, args=(modelname, ), axis=1)

    g = sns.barplot(df, x='length', y='model_cor',  hue="label", alpha=0.6)

    #g.set_axis_labels("Length", "Accuracy")
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels = ['Words', 'Passwords'])

    if plottitle is not None:
        title = plottitle
    else:
        title = modelname

    g.set(xlabel ="Length", ylabel = "Accuracy", title=title)

    file = filename.split("/")[-1]
    plt.savefig(f'failureanalysis/{file[:-4]}-{modelname}.png', bbox_inches='tight')
    '''

def analyze_file_for_totals_length(filename, xmax=20, plottitle=None):
    df = pd.read_csv(filename, index_col=0)
    df['length'] = df['text'].apply(lambda x: len(str(x)))
    print(df)

    # Filter out lengths higher than xmax to keep plot feasible (otherwise x-axis is to large)
    df = df[df['length'] <= xmax]

    g = sns.countplot(df, x='length', hue="label", alpha=0.6)

    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels = ['Words', 'Passwords'])

    if plottitle is not None:
        title = plottitle
    else:
        title = filename

    g.set(xlabel ="Length", ylabel = "Count", title=title)

    file = filename.split("/")[-1]
    plt.savefig(f'failureanalysis/{file[:-4]}-totals.png', bbox_inches='tight')

#analyze_file_for_length("./predictions/most_common_En1.0_500000_split2.csv", modelname='KNearestNeighborsminkowskiModel--most_common_En1.0_500000_split2-', plottitle="KNN Minkowski (size 500000, split 2)")
analyze_file_for_totals_length("./predictions/most_common_Ar1.0_10000_wordsonly.csv", plottitle="totals Arabic")
#analyze_file_for_lowercase("./predictionsSepFiles/predictionsBigrams/most_common_En1.0_1000")#_split2.csv")
