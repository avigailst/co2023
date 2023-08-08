from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np


class makeBagOfWord:
    def __int__(self ):
        return


    def make_bag(self, word_df, doc_df, file_name):
        word_len = len(word_df.index)
        bag_of_words = np.zeros((len(doc_df), word_len+1))

        for ind in doc_df.index:
            split_doc = doc_df['text'][ind].split(' ')
            vector = [0 for i in (range(word_len + 1))]
            for w in split_doc:
                filter = word_df['word'] == w
                if (filter).any():
                    index = word_df[filter].index[0]
                    vector[index] += 1
                vector[-1] = 0 if doc_df['task_1'][ind] == 'NOT' else 1
            bag_of_words = bag_of_words + [vector]
            data = pd.DataFrame(bag_of_words)
            data.to_csv(file_name)
        return np.array(data)






