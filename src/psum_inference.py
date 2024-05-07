import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from psum import PSUMTwoTasksClassifier
from Mawqifdataset import MawqifDataset

def model_inference(model, test_data, batch_size=16):

  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  if use_cuda:
    model = model.cuda()

  preds = [[], []]
  with torch.no_grad():

    for test_input in test_dataloader:
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)

      output = model(input_id, mask)
      
      agg1, agg2 = model.aggregate(output)
      preds[0] += agg1.argmax(dim=1).to("cpu").tolist()
      preds[1] += agg2.argmax(dim=1).to("cpu").tolist()

  return preds[0]


def psum_inference(test_df, model_path="models/psum_twotasks.pth", model_name="UBC-NLP/MARBERT"):
  model = PSUMTwoTasksClassifier(model_name,n_layers=4, n_classes_1=3, n_classes_2=3)
  for param in model.bert.parameters():
    param.requires_grad = False
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  test_dataset = MawqifDataset(test_df, tokenizer, model_name, task='both', is_inference=True, add_target=True)
  model = torch.load(model_path)
  model.eval()
  preds = model_inference(model, test_dataset)
  stance_to_int = {
    "AGAINST": 1,
    "FAVOR": 2,
    "NONE": 0
  }
  int_to_stance = {value: key for key, value in stance_to_int.items()}
  return [int_to_stance[p] for p in preds]