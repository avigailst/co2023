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


if_train = False
languages = ["Assamese", "Bangli", "Bodo"]
languages_add = ["A", "BE", "BO"]
makeBagOfWord = makeBagOfWord()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for ind, lang in enumerate(languages):
        print(lang)
        data_file_name = lang + "/train_" + languages_add[ind] + "_AH_HASOC2023.csv"
        test_file_name = lang + "/test_" + languages_add[ind] + "_AH_HASOC2023.csv"
        data = pd.read_csv(data_file_name)
        data = data.sample(frac=1).reset_index(drop=True)
        data_len = len(data.index)
        data_train = data
        data_test = data
        tf_idf_file_name = "tf_idf/tf_idf_"+lang+".csv"
        wordListA = WordListArange(tf_idf_file_name, data_train)
        word_list = wordListA.makeMatrix()
        word_list = word_list.drop(word_list[word_list['doc_count'] < 1].index)
        word_list.reset_index(drop=True, inplace=True)
        word_len = len(word_list.index)
        print(word_len)
        if if_train:
            for i in range(500,word_len, 500 ):
                train_bag = makeBagOfWord.make_bag(word_list[:i], data_train, False)
                x_train = train_bag[: , :-1]
                y_train = train_bag[ : , -1]
                test_bag =makeBagOfWord.make_bag(word_list[:i], data_test, False)
                x_test = test_bag[:, :-1]
                y_test = test_bag[:, -1]
                model = MultinomialNB(force_alpha=True)
                model.fit(x_train, y_train)
                # filename = 'finalized_model.sav'
                # pickle.dump(model, open(filename, 'wb'))
                y_pred = model.predict(test_bag[: , :-1])
                f1 = f1_score(y_test,y_pred , average='macro')
                # writer.writerow()
                print("MultinomialNB," + str(i) + ", " + str(f1))
                model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                model.fit(x_train, y_train)
                y_pred = model.predict(test_bag[:, :-1])
                f1 = f1_score(y_test, y_pred, average='macro')
                print("SVC," + str(i) + ", " + str(f1))
                model = RandomForestClassifier(max_depth=2, random_state=0)
                model.fit(x_train, y_train)
                y_pred = model.predict(test_bag[:, :-1])
                f1 = f1_score(y_test, y_pred, average='macro')
                print("RF," + str(i) + ", " + str(f1))
                model = LogisticRegression(random_state=0)
                model.fit(x_train, y_train)
                y_pred = model.predict(test_bag[:, :-1])
                f1 = f1_score(y_test, y_pred, average='macro')
                print("LR," + str(i) + ", " + str(f1))
                model = MLPClassifier(random_state=1, max_iter=300)
                model.fit(x_train, y_train)
                y_pred = model.predict(test_bag[:, :-1])
                f1 = f1_score(y_test, y_pred, average='macro')
                print("MLP," + str(i) + ", " + str(f1))
        else:
            train_bag = makeBagOfWord.make_bag(word_list, data ,False)
            x_train = train_bag[:, :-1]
            y_train = train_bag[:, -1]
            for mod in ["MultinomialNB", "SVC", "LR", "MLP"]:
                if mod == "MultinomialNB":
                    model = MultinomialNB(force_alpha=True)  # MultinomialNB
                elif mod == "SVC":
                    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))  # SVC
                elif mod == "LR":
                    model = LogisticRegression(random_state=0)  # LR
                else:
                    model = MLPClassifier(random_state=1, max_iter=300)  # MLP
                model.fit(x_train, y_train)
                test_data = pd.read_csv(test_file_name)
                test_bag = makeBagOfWord.make_bag(word_list, test_data,True)
                y_pred = model.predict(test_bag)
                text_pred = pd.DataFrame(data=y_pred, columns=['task_1'])
                text_pred.index = text_pred.index + 1
                text_pred.replace({'task_1': {0: 'NOT', 1: 'HOF'}}, regex=True, inplace=True)
                text_pred.to_csv('wordgram_allword_'+mod+ "_" +lang+'.csv', index_label='S. No.')






