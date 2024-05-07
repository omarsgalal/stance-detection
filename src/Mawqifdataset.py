import numpy as np
import transformers
import torch
from torch.optim import Adam
from arabert.preprocess import ArabertPreprocessor


class MawqifDataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, model_name, task='stance', is_inference=False, add_target=False):
    self.labelsIds = {'Neutral': 0, 'Against': 1, 'Favor': 2}
    self.sent_labels = {'Neutral': 0, 'Negative': 1, 'Positive': 2}
    self.task = task
    self.is_inference = is_inference
    if not self.is_inference:
      if task == 'stance':
        self.labels = [self.labelsIds[label] for label in df['stance']]
      elif task == 'both':
        self.labels = [[self.labelsIds[label] for label in df['stance']], [self.sent_labels[label] for label in df['sentiment']]]
    self.targets = df['target'].tolist()
    if model_name in ["aubmindlab/bert-base-arabertv02-twitter", "aubmindlab/bert-base-arabertv2"]:
      arabert_prep = ArabertPreprocessor(model_name=model_name)
      texts = [arabert_prep.preprocess(t) for t in df['text'].tolist()]
    else:
      texts = df['text'].tolist()
    if add_target:
      self.texts = [tokenizer(self.targets[i], text, padding='max_length', max_length = 128, truncation=True, return_tensors="pt") for i, text in enumerate(texts)]
    else:
      self.texts = [tokenizer(text, padding='max_length', max_length = 128, truncation=True, return_tensors="pt") for text in texts]

  def classes(self):
    return self.labels

  def __len__(self):
    return len(self.texts)

  def get_batch_labels(self, idx):
    if self.task == 'both':
        return np.array([self.labels[0][idx], self.labels[1][idx]])
    return np.array(self.labels[idx])

  def get_batch_texts(self, idx):
    return self.texts[idx]

  def __getitem__(self, idx):
    batch_texts = self.get_batch_texts(idx)
    if self.is_inference:
        return batch_texts
    batch_y = self.get_batch_labels(idx)
    return batch_texts, batch_y