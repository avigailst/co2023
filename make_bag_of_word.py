from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np


class makeBagOfWord:
    def __int__(self ):
        return

    def make_bag(self,word_data, doc_data, file_name):
        word_len = len(word_data.index)
        bag_of_words = []
        for ind in doc_data.index:
            split_doc = doc_data['text'][ind].split(' ')
            vector = [0 for i in (range(word_len + 1))]
            for w in split_doc:
                filter = word_data['word'] == w
                if (filter).any():
                    index = word_data[filter].index[0]
                    vector[index] += 1
                vector[-1] = 0 if doc_data['task_1'][ind] == 'NOT' else 1
            bag_of_words = bag_of_words + [vector]
            data = pd.DataFrame(bag_of_words)
            data.to_csv(file_name)
        return np.array(data)


print("")



