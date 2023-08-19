import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from word_list_arange import WordListArange
from make_bag_of_word import makeBagOfWord
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import pandas as pd
import torch
from collections import OrderedDict
import torch.nn as nn

languages = ["Assamese", "Bangli", "Bodo"]
languages_add = ["A", "BE", "BO"]

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
model = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertModel.from_pretrained("bert-base-multilingual-cased")
for tr in ["train", "test"]:
    print(tr)
    for ind, lang in enumerate(languages):
        print(lang)
        data_file_name = lang +  "/"+ tr +"_"+ languages_add[ind] + "_AH_HASOC2023.csv"
        data = pd.read_csv(data_file_name)
        data_len = len(data.index)
        bert_data = np.zeros(shape=(data_len, bert_len))
        task = np.zeros(shape=(data_len, 1))
        for ind, doc in enumerate(data['text']):
            # print(ind)
            encoded_input = tokenizer(doc, padding=True, truncation=True, return_tensors='pt')
            output = model(**encoded_input)
            embedding = output[0][:,0,:].detach().numpy()
            bert_data[ind] = embedding[0]
            if tr=="train":
                task[ind] = 0 if data['task_1'][ind]=='NOT' else 1

        data = pd.DataFrame(data=bert_data)
        if tr == "train":
            task = pd.DataFrame(data=task)
            data['task_1'] = task

        data.to_csv('bert_data_'+ tr+ "_" + lang +'.csv')
