import numpy as np
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModel
import transformers
import torch
from torch import nn
import pickle
# from arabert.preprocess import ArabertPreprocessor
from Mawqifdataset import MawqifDataset
stance_to_int = {
  "AGAINST": 0,
  "FAVOR": 1,
  "NONE": 2
}
int_to_stance = {value: key for key, value in stance_to_int.items()}

class AvgPoolerClassifier(nn.Module):
  def __init__(self, model_name= "UBC-NLP/MARBERT"):
    super(AvgPoolerClassifier, self).__init__()
    self.bert = AutoModel.from_pretrained(model_name)

  def forward(self, input_id, mask):
    last_hidden_state, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
    full_mask = mask.reshape(-1,128,1).repeat(1,1,768)
    last_hidden_state = last_hidden_state * full_mask
    last_hidden_state = last_hidden_state.sum(dim=1)
    last_hidden_state /= mask.sum(dim=2)
    return last_hidden_state

def getOutputEmbeddings(dataset_df, batch_size=1024, model_name="UBC-NLP/MARBERT"):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  test_dataset = MawqifDataset(dataset_df, tokenizer, model_name, task='stance', is_inference=True)

  dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

  model = AvgPoolerClassifier(model_name)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  if use_cuda:
    model = model.cuda()

  embeddings = []
  with torch.no_grad():
    for input in dataloader:
      mask = input['attention_mask'].to(device)
      input_id = input['input_ids'].squeeze(1).to(device)
      output = model(input_id, mask)
      tempEmbedding = output.detach().to('cpu').numpy()
      embeddings.append(tempEmbedding)
  return np.concatenate(embeddings, axis=0)

def avg_pooler_inference(dataset_df, model_path="models/arabertv02_LOGREG_MODEL.pkl", embedding_model="UBC-NLP/MARBERT"):
    embeddings = getOutputEmbeddings(dataset_df, model_name=embedding_model)

    # TODO: pass embeddings to the model
    # The same code can be used for both models
    with open(model_path, "rb") as fin:
        model = pickle.load(fin)
    y_pred = model.predict(embeddings)
    y_pred = [int_to_stance[i] for i in y_pred]
    return y_pred
    