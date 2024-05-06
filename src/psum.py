from transformers import AutoConfig, AutoModel
import transformers
from torch import nn

class PSUMTwoTasksClassifier(nn.Module):
  def __init__(self, model_name= "UBC-NLP/MARBERT",n_layers=4, n_classes_1=3, n_classes_2=3, max_length=128, separate_bert_layers_for_tasks=False, informed_by='', one_softmax_informing=False, detach=False):
    super(PSUMTwoTasksClassifier, self).__init__()
    self.config = AutoConfig.from_pretrained(model_name)

    self.bert = AutoModel.from_pretrained(model_name)

    self.bertLayers = nn.ModuleList()
    if separate_bert_layers_for_tasks:
      self.bertLayers2 = nn.ModuleList()
    self.linears_1 = nn.ModuleList()
    self.linears_2 = nn.ModuleList()
    self.n_layers = n_layers
    self.n_classes_1 = n_classes_1
    self.n_classes_2 = n_classes_2
    self.separate_bert_layers_for_tasks = separate_bert_layers_for_tasks
    self.all_informed = informed_by != ''
    self.informed_by = informed_by
    self.one_softmax_informing = one_softmax_informing
    self.detach = detach

    if self.all_informed:
      self.softmaxes_1 = nn.ModuleList()
      self.softmaxes_2 = nn.ModuleList()
      self.informed_linears_1 = nn.ModuleList()
      self.informed_linears_2 = nn.ModuleList()


    for i in range(n_layers):
      self.bertLayers.append(transformers.BertLayer(self.config))
      if separate_bert_layers_for_tasks:
        self.bertLayers2.append(transformers.BertLayer(self.config))
      self.linears_1.append(nn.Linear(768, n_classes_1))
      self.linears_2.append(nn.Linear(768, n_classes_2))

      if self.all_informed:
        self.softmaxes_1.append(nn.Softmax(dim=1))
        self.softmaxes_2.append(nn.Softmax(dim=1))
        self.informed_linears_1.append(nn.Linear(768 + n_classes_2, n_classes_1))
        self.informed_linears_2.append(nn.Linear(768 + n_classes_1, n_classes_2))


  def forward(self, input_id, mask):
    hidden_states = self.bert(input_ids= input_id, attention_mask=mask, return_dict=True, output_hidden_states=True)['hidden_states']
    final_outputs_1 = []
    final_outputs_2 = []

    if self.all_informed:
      informed_final_outputs_1 = []
      informed_final_outputs_2 = []

    for i in range(self.n_layers):
      final_outputs_1.append(self.linears_1[i](self.bertLayers[i](hidden_states[-i-1])[0][:,0,:]))
      if self.separate_bert_layers_for_tasks:
        final_outputs_2.append(self.linears_2[i](self.bertLayers2[i](hidden_states[-i-1])[0][:,0,:]))
      else:
        final_outputs_2.append(self.linears_2[i](self.bertLayers[i](hidden_states[-i-1])[0][:,0,:]))

      if self.all_informed and not self.one_softmax_informing:
        if self.detach:
          fo2i = torch.Tensor.detach(final_outputs_2[i])
          fo1i = torch.Tensor.detach(final_outputs_1[i])
        else:
          fo2i = final_outputs_2[i]
          fo1i = final_outputs_1[i]

        informed_final_outputs_1.append(self.informed_linears_1[i](torch.cat((self.bertLayers[i](hidden_states[-i-1])[0][:,0,:], self.softmaxes_2[i](fo2i)), dim=1)))
        informed_final_outputs_2.append(self.informed_linears_2[i](torch.cat((self.bertLayers[i](hidden_states[-i-1])[0][:,0,:], self.softmaxes_1[i](fo1i)), dim=1)))

    if self.all_informed and self.one_softmax_informing:
      sum1 = final_outputs_1[0]
      sum2 = final_outputs_2[0]
      for j in range(1, self.n_layers):
        sum1 += final_outputs_1[j]
        sum2 += final_outputs_2[j]
      sum1 /= self.n_layers
      sum2 /= self.n_layers

      o1 = self.softmaxes_1[0](sum1)
      o2 = self.softmaxes_2[0](sum2)

      for i in range(self.n_layers):
        informed_final_outputs_1.append(self.informed_linears_1[i](torch.cat((self.bertLayers[i](hidden_states[-i-1])[0][:,0,:], o2), dim=1)))
        informed_final_outputs_2.append(self.informed_linears_2[i](torch.cat((self.bertLayers[i](hidden_states[-i-1])[0][:,0,:], o1), dim=1)))

    if self.all_informed:
      return final_outputs_1, final_outputs_2, informed_final_outputs_1, informed_final_outputs_2

    return final_outputs_1, final_outputs_2

  def loss(self, output, labels, criterion):
    bloss = 0.0
    for i in range(self.n_layers):
      if self.informed_by in ['', 't1', 'all']:
        bloss += criterion(output[0][i], labels[:, 0].long())

      if self.informed_by in ['', 't2', 'all']:
        bloss += criterion(output[1][i], labels[:, 1].long())

      if self.informed_by in ['t2', 'all']:
        bloss += criterion(output[2][i], labels[:, 0].long())

      if self.informed_by in ['t1', 'all']:
        bloss += criterion(output[3][i], labels[:, 1].long())

    return bloss

  def aggregate(self, output):
    if self.informed_by == '':
      t1_idx = 0
      t2_idx = 1
    elif self.informed_by == 't1':
      t1_idx = 0
      t2_idx = 3
    elif self.informed_by == 't2':
      t1_idx = 2
      t2_idx = 1
    elif self.informed_by == 'all':
      t1_idx = 2
      t2_idx = 3

    agg1 = output[t1_idx][0]
    agg2 = output[t2_idx][0]
    for i in range(1, len(output[0])):
      agg1 += output[t1_idx][i]
      agg2 += output[t2_idx][i]
    return agg1 / len(output[t1_idx]), agg2 / len(output[t2_idx])

  def calcAcc(self, output, labels):
    return (self.aggregate(output)[0].argmax(dim=1) == labels[:, 0]).sum().item()

  def evaluate(self, output, test_label, preds, golds):
    if len(preds) == 0:
      preds = [[], []]
      golds = [[], []]

    agg1, agg2 = self.aggregate(output)
    preds[0] += agg1.argmax(dim=1).to("cpu").tolist()
    golds[0] += test_label[:, 0].to("cpu").tolist()
    preds[1] += agg2.argmax(dim=1).to("cpu").tolist()
    golds[1] += test_label[:, 1].to("cpu").tolist()

    return preds, golds

  def evaluation_report(self, preds, golds, is_f1pn=False):
    print("########## Task 1 Results ###############")
    print(classification_report(golds[0], preds[0], digits=4))
    if self.n_classes_1 == 3:
      r = classification_report(golds[0], preds[0], digits=4, output_dict=True)
      f1pn = (r["1"]["f1-score"] + r["2"]["f1-score"]) / 2.0
      print(f'F1PN: {f1pn:.4f}')

    print("\n\n########## Task 2 Results ###############")
    print(classification_report(golds[1], preds[1], digits=4))
    if self.n_classes_2 == 3:
      r = classification_report(golds[1], preds[1], digits=4, output_dict=True)
      f1pn = (r["1"]["f1-score"] + r["2"]["f1-score"]) / 2.0
      print(f'F1PN: {f1pn:.4f}')