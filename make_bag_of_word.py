from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np


class makeBagOfWord:
    def __int__(self ):
        return


    def make_bag(self, word_df, doc_df,if_test):
        word_len = len(word_df.index)
        if not if_test:
            word_len += 1
        bag_of_words = np.zeros((len(doc_df), word_len))
        for i, ind in enumerate(doc_df.index):
            split_doc = doc_df['text'][ind].split(' ')
            for w in split_doc:
                filter = word_df['word'] == w
                if (filter).any():
                    index = word_df[filter].index[0]
                    bag_of_words[i, index] += 1
            if not if_test:
                bag_of_words[i, -1] = 0 if doc_df['task_1'][ind] == 'NOT' else 1

            # data = pd.DataFrame(bag_of_words)
            # data.to_csv(file_name)
        return bag_of_words

    def make_bag_ngram(self, ngram_df, doc_df,if_test,n):
        ngram_len = len(ngram_df.index)
        if not if_test:
            ngram_len += 1
        bag_of_words = np.zeros((len(doc_df), ngram_len))
        for i, ind in enumerate(doc_df.index):
            # print(i)
            sentence_ngrams = [doc_df['text'][ind][i:i + n] for i in range(len(doc_df['text'][ind]) - n + 1)]
            for w in sentence_ngrams:
                filter = ngram_df['ngram'] == w
                if (filter).any():
                    index = ngram_df[filter].index[0]
                    bag_of_words[i, index] += 1
            if not if_test:
                bag_of_words[i, -1] = 0 if doc_df['task_1'][ind] == 'NOT' else 1

            # data = pd.DataFrame(bag_of_words)
            # data.to_csv(file_name)
        return bag_of_words





