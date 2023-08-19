import torch
from collections import OrderedDict
import torch.nn as nn
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

bert_len = 768

language = "Assames"
# language = "Bangli"
# language = "Bodo"
bert_len= 768


data_file_name = 'bert_data_'+ language+'.csv'
data_test_file_name = 'bert_data_test_'+ language+'.csv'
if __name__ == '__main__':

    data = pd.read_csv(data_file_name, index_col=0)
    columns = data.iloc[0]
    data.columns = columns
    data.index.name = None
    data = data[1:]
    data_len = len(data.index)
    data = data.sample(frac=1).reset_index(drop=True)
    data_train = data.to_numpy()
    x_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    model = MultinomialNB(force_alpha=True)  # MultinomialNB
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    f1 = f1_score(y_test, y_pred, average='macro')
    print("NB," + str(i) + ", " + str(f1))


    data_test = pd.read_csv(data_test_file_name, index_col=0)
    columns = data_test.iloc[0]
    data_test.columns = columns
    data_test.index.name = None
    data_test = data_test.to_numpy()
    y_pred = model.predict()
    y_pred = model.predict(data_test)
    text_pred = pd.DataFrame(data=y_pred, columns=['task_1'])
    # text_pred.index += 1
    # text_pred.replace({'task_1' : {0 : 'NOT', 1 : 'HOF'}}, regex=True)
    text_pred.to_csv('test_bert_'+language+'_NB.csv', index_label='S. No.')
    




    # Process is complete.
    print('Training process has finished.')

