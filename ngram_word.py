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
from word_list_arange import WordListArange

def make_ngram(sentences, n):
    sentence_ngrams = [sentence[i:i + n] for sentence in sentences for i in range(len(sentence) - n + 1)]
    # Create a set of unique n-grams across all sentences
    all_ngrams = list(set(sentence_ngrams))

    # Create a table structure to store n-gram presence
    ngram_table = pd.DataFrame(index=sentences, columns=all_ngrams, dtype=int)
    return ngram_table



def generate_ngram_table(sentences, n):
    # Generate n-grams for each sentence
    ngram_table.fillna(0, inplace=True)

    # Populate the table with n-gram presence values
    for i, sentence in enumerate(sentences):
        for j in range(len(sentence) - n + 1):
            ngram = sentence[j:j + n]
            ngram_table.at[sentences[i], ngram] = 1

    return ngram_table
languages = ["Assamese", "Bangli", "Bodo"]
languages_add = ["A", "BE", "BO"]

makeBagOfWord = makeBagOfWord()
tr = "test"
if __name__ == '__main__':
   for n in [5, 6]:
    for ind, lang in enumerate(languages):
        print(lang)
        data_file_name = lang + "/train_" + languages_add[ind] + "_AH_HASOC2023.csv"
        test_file_name = lang + "/test_" + languages_add[ind] + "_AH_HASOC2023.csv"
        data = pd.read_csv(data_file_name)
        data = data.sample(frac=1).reset_index(drop=True)
        data_len = len(data.index)
        data_file_name = 'tf_idf_'+str(n)+'_gram_'+ lang +'.csv'
        ngram_list = pd.read_csv(data_file_name)
        ngram_list = ngram_list.drop(ngram_list[ngram_list['doc_count'] < 2].index)
        ngram_list.reset_index(drop=True, inplace=True)
        ngram_len = len(ngram_list.index)
        print(ngram_len)
        tf_idf_file_name = "tf_idf/tf_idf_" + lang + ".csv"
        wordListA = WordListArange(tf_idf_file_name, data)
        word_list = wordListA.makeMatrix()
        word_list = word_list.drop(word_list[word_list['doc_count'] < 1].index)
        word_list.reset_index(drop=True, inplace=True)
        train_bag_ngram = makeBagOfWord.make_bag_ngram(ngram_list, data, False, n)
        train_bag_word = makeBagOfWord.make_bag(word_list, data ,False)
        x_train_word = np.array(train_bag_word[:, :-1])
        x_train_ngram = np.array(train_bag_ngram[:, :-1])
        x_train = np.concatenate((x_train_ngram, x_train_word), axis= 1)
        y_train = train_bag_ngram[:, -1]
        test_data = pd.read_csv(test_file_name)
        test_bag_ngram = makeBagOfWord.make_bag_ngram(ngram_list, test_data, True, n)
        test_bag_word = makeBagOfWord.make_bag(word_list, test_data, True)
        x_test = np.concatenate((test_bag_ngram, test_bag_word), axis=1)
        for mod in ["MLP"]: #["MultinomialNB","LR"]:
            if mod=="MultinomialNB":
                model = MultinomialNB(force_alpha=True) # MultinomialNB
            elif mod=="LR":
                model = LogisticRegression(random_state=0) #LR
            else:
                model = MLPClassifier(random_state=1, max_iter=500)  # MLP
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            text_pred = pd.DataFrame(data=y_pred, columns=['task_1'])
            text_pred.index = text_pred.index + 1
            text_pred.replace({'task_1' : {0 : 'NOT', 1 : 'HOF'}}, regex=True, inplace=True)
            test_file_name = 'test_gte2_ngram_'+str(n)+"_"+ lang+ '_'+mod +'_word_gte2.csv'
            text_pred.to_csv(test_file_name, index_label='S. No.')
