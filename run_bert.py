import torch
from collections import OrderedDict
import torch.nn as nn
import pandas as pd

bert_len = 768
language = "Assames"
# language = "Bangli"
# language = "Bodo"
languages = ["Assamese", "Bangli", "Bodo"]
def macro_f1(y_pred, y_true):
    """Compute the macro F1 score.

    Args:
        y_pred (torch.Tensor): Predicted labels, shape (N, C) where C is number of classes
        y_true (torch.Tensor): Ground truth labels, shape (N, C)

    Returns:
        torch.Tensor: Macro F1 score.
    """

    assert y_pred.shape == y_true.shape

    tp = (y_true * y_pred).sum(dim=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    macro_f1 = f1.mean()
    return macro_f1

if __name__ == '__main__':
    languages = ["Assamese", "Bangli", "Bodo"]
    languages_add = ["A", "BE", "BO"]


    for ind, ang in enumerate(languages):
        print(ang)
        data_train_file_name= 'bert_data_train_' + ang +'.csv'
        data_test_file_name = 'bert_data_test_' + ang + '.csv'
        data = pd.read_csv(data_train_file_name, index_col=0)
        columns = data.iloc[0]
        data.columns = columns
        data.index.name = None
        data = data[1:]
        data_len = len(data.index)
        data = data.sample(frac=1).reset_index(drop=True)
        data_train = data.to_numpy() #data[:int(data_len * 0.7)].to_numpy()
        # data_test = data[int(data_len * 0.7):].to_numpy()
        model =  nn.Sequential(
            nn.Linear(bert_len, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Run the training loop
        for epoch in range(0, 2000):

            # Print epoch
            print(f'Starting epoch {epoch + 1}')


            inputs = torch.tensor(data_train[:, :-1], dtype=torch.float32)
            targets = torch.tensor(data_train[:, -1], dtype=torch.float32).view(-1, 1)

            y_pred = model(inputs)
            loss = criterion(y_pred, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch%10 == 0:
                f1 = macro_f1(y_pred, targets)
                print('F1  %.3f' %  f1)

        data_test = pd.read_csv(data_test_file_name, index_col=0)
        columns = data_test.iloc[0]
        data_test.columns = columns
        data_test.index.name = None
        data_test = data_test.to_numpy()
        inputs = torch.tensor(data_test, dtype=torch.float32)
        y_pred = model(inputs)
        y_pred = y_pred.detach().numpy()
        y_pred[y_pred>=0.5] =1
        y_pred[y_pred< 0.5] = 0
        text_pred = pd.DataFrame(data=y_pred, columns=['task_1'])

        text_pred.to_csv('test_bert_2_'+ang+'.csv', index_label='S. No.')



        # Process is complete.
        print('Training process has finished.')

