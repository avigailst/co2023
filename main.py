from word_list_arange import WordListArange
from make_bag_of_word import makeBagOfWord
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import pandas as pd
import csv

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_file_name = "Assamese/train_A_AH_HASOC2023.csv"
    data = pd.read_csv(data_file_name)
    data = data.sample(frac=1).reset_index(drop=True)
    data_len = len(data.index)
    data_train = data[:int(data_len*0.7)]
    data_test = data[int(data_len*0.7): ]
    tf_idf_file_name = "tf_idf_assamese.csv"
    wordListA = WordListArange(tf_idf_file_name, data_train)
    makeBagOfWord = makeBagOfWord()
    word_list = wordListA.makeMatrix()
    word_list = word_list.drop(word_list[word_list['doc_count'] < 2].index)
    word_list.reset_index(drop=True, inplace=True)
    word_len = len(word_list.index)
    f = open('results.csv', 'w')
    writer = csv.writer(f)
    for i in range(500,word_len, 500 ):
        train_bag = makeBagOfWord.make_bag(word_list[:i], data_train, 'AH_train_'+ str(i))
        test_bag =makeBagOfWord.make_bag(word_list[:i], data_test, 'AH_test_'+ str(i))
        clf = MultinomialNB(force_alpha=True)
        y_pred= clf.predict(test_bag[::-1])
        f1 = f1_score(test_bag[:-1:], y_pred, average='macro')
        # writer.writerow()
        print(str(i) + ", " + str(f1))




