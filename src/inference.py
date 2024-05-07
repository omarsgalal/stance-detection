import numpy as np
import pandas as pd
from psum_inference import psum_inference
from psum import PSUMTwoTasksClassifier
from avg_pooler import avg_pooler_inference
from ensemble import majority_voting
from writer import save_output
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Predict stance of a dataset of tweets with their targets')

# Add arguments
parser.add_argument('--dataset_path', type=str, help='Path to the csv dataset', default="datasets/Mawqif_AllTargets_Blind Test.csv")
parser.add_argument('--output_path', type=str, help='Path to the standard format output file', default="predictions.csv")

# Parse the command-line arguments
args = parser.parse_args()

# load dataset
test_df = pd.read_csv(args.dataset_path)

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

# save output
test_df['stance'] = vote_preds
save_output(test_df, args.output_path)