from nltk.tokenize import word_tokenize
import math
import pandas as pd
import numpy as np
import scipy.sparse
from sinling import SinhalaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def makeMatrix(file_name):
    # make a set of words in all files(for the columns)
    # Assuming the "text" column contains the non-English text data
    data = pd.read_csv(file_name)
    docs = data["text"].tolist()

    # Initialize the TF-IDF vectorizer with the appropriate language
    # sin_tokenizer = SinhalaTokenizer()

    # # Tokenize the tweets to get the list of word lists
    # tokenized_docs = [sin_tokenizer.tokenize(text) for text in docs]

    # # Create a set of all unique words in the dataset
    # unique_words = set(word for tweet in tokenized_docs for word in docs)

    # Create a set of all unique words in the dataset
    unique_words = set()
    tokenized_docs = []
    for doc in docs:
        split_doc = doc.split(' ')
        tokenized_docs = tokenized_docs + [split_doc]
        unique_words = unique_words.union(set(split_doc))

    df_list = []
    for doc in docs:
        num_doc_words = []
        doc_words = doc.split(' ')
        num_doc_words = dict.fromkeys(unique_words, 0)

        # for TFIDF with BM25
        tf = Tf_calculation(num_doc_words, doc_words)

        # tf_idf_bm25 = TF_IDF_BM25_calculation(query, tf, doc, tokenized_docs, K_VALUE, B_VALUE)

        # df_list = df_list + [{**tf_idf_bm25}] #list of all dictonaries

        # for regular TFIDF
        tf_idf = TF_IDF_calculation(tf, tokenized_docs)
        df_list = df_list + [{**tf_idf}]  # list of all dictonaries

    df = pd.DataFrame(df_list)

    # converting to sparse matrix
    sdf = scipy.sparse.csr_matrix(df.values)

    return sdf


def Tf_calculation(documentDict, documentWords):
    tf_Dict = {}
    for word, count in documentDict.items():
        count = documentWords.count(word)
        tf_Dict[word] = (count / len(documentWords))
    return tf_Dict


def IDF_calculation(token, tokenized_docs):
    counter = sum([1 for doc in tokenized_docs if token in doc])
    print('token-', token, ' c-', counter)

    return counter


def TF_IDF_calculation(tfBagOfWords, tokenized_docs):
    tf_idf = {}
    for word, val in tfBagOfWords.items():
        idf = IDF_calculation(word, tokenized_docs)
        idfval = math.log(len(tokenized_docs) / float(idf + 1))
        tf_idf[word] = val * idfval
    return tf_idf


def average_document_length(docs):
    avg = sum([len(doc) for doc in docs]) / len(docs)
    return avg


if __name__ == '__main__':
    res = makeMatrix('Assamese/train_A_AH_HASOC2023.csv')
    # # Save the results to a new CSV file
    output_file = "output3.csv"
    res.to_csv(output_file, index=False)