import numpy as np
import pandas as pd
from psum_inference import psum_inference
from psum import PSUMTwoTasksClassifier
from avg_pooler import avg_pooler_inference

# load dataset
test_df = pd.read_csv("datasets/Mawqif_AllTargets_Blind Test.csv")

# Inference with psum
psum_preds = psum_inference(test_df, model_path="models/psum_twotasks.pth")

# Inference with avg pooler
avg_pooler_preds = avg_pooler_inference(test_df)

print(len(psum_preds))
print(avg_pooler_preds.shape)