import numpy as np
import pandas as pd
from psum_inference import psum_inference
from psum import PSUMTwoTasksClassifier
from avg_pooler import avg_pooler_inference
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from ensemble import majority_voting
from avg_pooler import stance_to_int
from sklearn.model_selection import train_test_split
from writer import save_output

def evaluate_official(test_df, y_pred):
    sum_f2_final = 0
    sum_f3_final = 0
    results = {}
    stance_to_int_gold = {
        "Against": 0,
        "Favor": 1,
        "Neutral": 2
    }
    test_df['stance_int'] = test_df['stance'].map(stance_to_int_gold)
    y_pred = [stance_to_int[p] for p in y_pred]
    for target in test_df["target"].unique():
        target_indices = [i for i in range(len(test_df['target'].tolist())) if test_df['target'].tolist()[i] == target]
        filtered_test_labels = [test_df['stance_int'].tolist()[i] for i in target_indices]
        filtered_predictions = [y_pred[i] for i in target_indices]
        # print(classification_report(filtered_test_labels, filtered_predictions))
        f1_3class = f1_score(filtered_test_labels, filtered_predictions, average = None)
        sum_f2_final += (f1_3class[0] + f1_3class[1])/2
        sum_f3_final += sum(f1_3class)/3
        results[target] = {"F1_score_2class": (f1_3class[0] + f1_3class[1])/2, "F1_score_3class": sum(f1_3class)/3}
    results["All Targets"] = {"F1_score_2class": sum_f2_final/3, "F1_score_3class": sum_f3_final/3}
    return results

# load dataset
dataset = load_dataset("NoraAlt/Mawqif_Stance-Detection")

# convert to pandas dataframe
df = pd.DataFrame({k: dataset['train'][k] for k, _ in dataset['train'].features.items()})
df['stance'] = df['stance'].apply(lambda x: "Neutral" if x is None else x)

# train test split
train_df, test_df = train_test_split(df, test_size=500, random_state=12345)

# Inference with psum
psum_preds = psum_inference(test_df, model_path="models/psum_twotasks.pth")

# Inference with psum
# psum_preds_2 = psum_inference(test_df, model_path="models/psum_twotasks_arabert.pth", model_name="aubmindlab/bert-base-arabertv02-twitter")

# Inference with avg pooler 1
avg_pooler_preds_1 = avg_pooler_inference(test_df, "models/MARBERT_LOGREG_MODEL.pkl", "UBC-NLP/MARBERT")

# Inference with avg pooler 1
avg_pooler_preds_2 = avg_pooler_inference(test_df, "models/arabertv02_LOGREG_MODEL.pkl", "aubmindlab/bert-base-arabertv02-twitter")

# majority vote predictions
vote_preds = majority_voting([psum_preds, avg_pooler_preds_1, avg_pooler_preds_2])

print(evaluate_official(test_df, psum_preds))
# print(evaluate_official(test_df, psum_preds_2))
print(evaluate_official(test_df, avg_pooler_preds_1))
print(evaluate_official(test_df, avg_pooler_preds_2))
print(evaluate_official(test_df, vote_preds))

save_output(test_df, 'gold.tsv')
pred_df = test_df.copy()
pred_df['stance'] = vote_preds
save_output(pred_df, 'predictions.tsv')