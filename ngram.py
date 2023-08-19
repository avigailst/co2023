import pandas as pd
import numpy as np
from word_list_arange import WordListArange
from make_bag_of_word import makeBagOfWord
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import pandas as pd
import csv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def make_ngram(sentences, n):
    sentence_ngrams = [sentence[i:i + n] for sentence in sentences for i in range(len(sentence) - n + 1)]
    # Create a set of unique n-grams across all sentences
    all_ngrams = list(set(sentence_ngrams))

    # Create a table structure to store n-gram presence
    ngram_table = pd.DataFrame(index=sentences, columns=all_ngrams, dtype=int)
    return ngram_table

def make_tf_idf(sentences, n):
    ngram_by_rows = pd.DataFrame()
    for ind in sentences.index:
        sentence = sentences['text'][ind]
        sentence_ngrams = [sentence[i:i + n] for i in range(len(sentence) - n + 1)]
        dd = [(a, 1) for a in sentence_ngrams]
        d1 = pd.DataFrame(data=dd, columns=['ngram', 'ngram_count']).groupby(['ngram']).sum().reset_index()
        d1['doc_count'] = [1 for i in range(len(d1))]
        d1['ngram_len'] = [len(sentence_ngrams) for i in range(len(d1))]
        add_row = [ngram_by_rows, d1]
        ngram_by_rows = pd.concat(add_row)
    data_arange = ngram_by_rows.groupby(['ngram']).sum().reset_index()
    data_arange.sort_values(by=['doc_count'])
    data_arange['ngram'].replace(to_replace='', value=np.nan, inplace=True)
    data_arange.dropna(subset=['ngram'], inplace=True)
    return data_arange

def generate_ngram_table(sentences, n):
    # Generate n-grams for each sentence
    ngram_table.fillna(0, inplace=True)

    # Populate the table with n-gram presence values
    for i, sentence in enumerate(sentences):
        for j in range(len(sentence) - n + 1):
            ngram = sentence[j:j + n]
            ngram_table.at[sentences[i], ngram] = 1

    return ngram_table
languages = ["Bangli"]#["Assamese", "Bangli", "Bodo"]
languages_add = ["BE"]#["A", "BE", "BO"]

makeBagOfWord = makeBagOfWord()
tr = "test"
if __name__ == '__main__':
   for n in [6]:
    for ind, lang in enumerate(languages):
        print(lang)
        data_file_name = lang + "/train_" + languages_add[ind] + "_AH_HASOC2023.csv"
        test_file_name = lang + "/test_" + languages_add[ind] + "_AH_HASOC2023.csv"
        data = pd.read_csv(data_file_name)
        data = data.sample(frac=1).reset_index(drop=True)
        data_len = len(data.index)
        data_to_write = make_tf_idf(data, n)
        data_to_write.to_csv('tf_idf_'+str(n)+'_gram_'+ lang +'.csv')
        data_file_name = 'tf_idf_'+str(n)+'_gram_'+ lang +'.csv'
        ngram_data = pd.read_csv(data_file_name)
        ngram_list = ngram_data.drop(ngram_data[ngram_data['doc_count'] < 1].index)
        ngram_list.reset_index(drop=True, inplace=True)
        ngram_len = len(ngram_list.index)
        print(ngram_len)
        if tr == "train":
            for i in range(2500,ngram_len, 2500 ):
                train_bag = makeBagOfWord.make_bag_ngram(ngram_list[:i], data, False,n)
                x_train = train_bag[: , :-1]
                y_train = train_bag[ : , -1]
                # test_bag =makeBagOfWord.make_bag(ngram_list[:i], data_test, False)
                # x_test = test_bag[:, :-1]
                # y_test = test_bag[:, -1]
                model = MultinomialNB(force_alpha=True)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_train)
                f1 = f1_score(y_train,y_pred , average='macro')

                print("MultinomialNB," + str(i) + ", " + str(f1))
                model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                model.fit(x_train, y_train)
                y_pred = model.predict(x_train)
                f1 = f1_score(y_train, y_pred, average='macro')
                print("SVC," + str(i) + ", " + str(f1))
                model = RandomForestClassifier(max_depth=2, random_state=0)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_train)
                f1 = f1_score(y_train, y_pred, average='macro')
                print("RF," + str(i) + ", " + str(f1))
                model = LogisticRegression(random_state=0)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_train)
                f1 = f1_score(y_train, y_pred, average='macro')
                print("LR," + str(i) + ", " + str(f1))
                model = MLPClassifier(random_state=1, max_iter=300)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_train)
                f1 = f1_score(y_train, y_pred, average='macro')
                print("MLP," + str(i) + ", " + str(f1))
        else:
            train_bag = makeBagOfWord.make_bag_ngram(ngram_list, data, False, n)
            x_train = train_bag[:, :-1]
            y_train = train_bag[:, -1]
            test_data = pd.read_csv(test_file_name)
            test_bag = makeBagOfWord.make_bag_ngram(ngram_list, test_data, True, n)
            for mod in ["MultinomialNB","SVC","LR", "MLP"]:
                if mod=="MultinomialNB":
                    model = MultinomialNB(force_alpha=True) # MultinomialNB
                elif mod=="SVC":
                    model = make_pipeline(StandardScaler(), SVC(gamma='auto')) #SVC
                elif mod=="LR":
                    model = LogisticRegression(random_state=0) #LR
                else:
                    model = MLPClassifier(random_state=1, max_iter=300)  # MLP
                model.fit(x_train, y_train)
                y_pred = model.predict(test_bag)
                text_pred = pd.DataFrame(data=y_pred, columns=['task_1'])
                text_pred.index = text_pred.index + 1
                text_pred.replace({'task_1' : {0 : 'NOT', 1 : 'HOF'}}, regex=True, inplace=True)
                test_file_name = 'test_ngram_'+str(n)+"_"+ lang+ '_'+mod +'.csv'
                text_pred.to_csv(test_file_name, index_label='S. No.')
