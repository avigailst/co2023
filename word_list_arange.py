from nltk.tokenize import word_tokenize
import math
import pandas as pd
import numpy as np
import scipy.sparse
from sinling import SinhalaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

class WordListArange:
    def __init__(self, file_name, data):
        self.file_name = file_name
        self.data = data

    def makeMatrix(self):
        # make a set of words in all files(for the columns)
        # Assuming the "text" column contains the non-English text data
        # data = pd.read_csv(self.in_file_name)

        word_by_rows = pd.DataFrame()
        for ind in self.data.index:
            split_doc = self.data['text'][ind].split(' ')
            dd = [(a,1) for a in split_doc]
            d1 = pd.DataFrame(data=dd, columns=['word', 'word_count']).groupby(['word']).sum().reset_index()
            d1['doc_count'] = [1 for i in range(len(d1))]
            d1['doc_len'] = [len(split_doc) for i in range(len(d1))]
            add_row = [word_by_rows, d1]
            word_by_rows = pd.concat(add_row)
        data_arange= word_by_rows.groupby(['word']).sum().reset_index()
        data_arange.sort_values(by=['doc_count'])
        data_arange['word'].replace(to_replace='', value=np.nan, inplace=True)
        data_arange.dropna(subset=['word'], inplace=True)
        # filepath = Path(self.file_name)
        # filepath.parent.mkdir(parents=True, exist_ok=True)
        # data_arange.to_csv(filepath)
        return data_arange


