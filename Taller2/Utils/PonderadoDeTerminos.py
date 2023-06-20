import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
print(os.path.dirname(__file__))
from .CustomLogger import CustomLogger
logger = CustomLogger(name='Ponderado de Terminos')


class TermWeighting():


    def compute_tfidf(self, texts):
        texts = self.__reconstruction(texts)
        logger.info('Starting the tfidf')
        logger.debug(f'The size of the texts is: {len(texts)}')
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(texts)

    def compute_binary_idf(self, texts):
        texts = self.__reconstruction(texts)
        logger.info('Starting the binary tfidf')
        logger.debug(f'The size of the texts is: {len(texts)}')
        vectorizer = CountVectorizer(binary=True)
        return vectorizer.fit_transform(texts)

    def normalize_by_length(self, texts):
        texts = self.__reconstruction(texts)
        logger.info('Normalizing by length')
        logger.debug(f'The size of the texts is: {len(texts)}')
        counts = self.__vectorize_and_count(texts)
        print(counts, type(counts))
        lengths = counts.sum(axis=1)
        return np.asarray(counts / lengths)

    def normalize_by_euclidean_distance(self, texts):
        texts = self.__reconstruction(texts)
        logger.info('Normalizing by euclidean distance')
        logger.debug(f'The size of the texts is: {len(texts)}')
        counts = self.__vectorize_and_count(texts)
        print(type(counts))
        if isinstance(counts, csr_matrix):
            # Si counts es una matriz dispersa, se utiliza multiply
            distances = np.sqrt((counts.multiply(counts)).sum(axis=1))
        else:
            # Si counts no es una matriz dispersa, se utiliza np.power o np.square
            distances = np.sqrt(np.sum(np.power(counts, 2), axis=1))
        return np.asarray(counts / distances)


    def __vectorize_and_count(self, texts):
        logger.info('Vectorizing and count')
        vectorizer = CountVectorizer()
        counts = vectorizer.fit_transform(texts)
        return np.asarray(counts)

    def __reconstruction(self, texts):
        logger.info('From a list of strings a string is created')
        texts = [' '.join(words) for words in texts]
        return texts