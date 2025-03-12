from sklearn.metrics import roc_auc_score
from utils import read_in_csv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(folder, lang, split, model_type):
    # Plot the ROC curve
    plt.figure()

    for size in ['1000', '10000', '100000', '500000']:
        df = read_in_csv(folder + lang + size + split + ".csv")
        # Calculate ROC curve
        col_name = model_type + "_" + lang + size + split + "-1"
        fpr, tpr, thresholds = roc_curve(df['label'], df[col_name]) 
        roc_auc = auc(fpr, tpr)
    
        plt.plot(fpr, tpr, label='Size ' + size + ' (area = %0.2f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_type.split("_")[0]}')
    plt.legend()
    plt.savefig(f'ROC Curve for {model_type.split("_")[0]}.png')

    #plt.scatter(fpr, tpr)
    #plt.show()

for model in ["RandomForest", "NaiveBayes", "MultinomialNaiveBayes", "KNearestNeighbors", "DecisionTree", "AdaBoost"]:
    plot_roc('./pred_proba/', 'most_common_En1.0_', '_split0', f'{model}Model')
