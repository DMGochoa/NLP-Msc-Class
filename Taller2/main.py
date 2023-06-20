import re
import os
import nltk
import spacy
import subprocess
import pandas as pd

from Utils.Combinatorio import CombinationOfMethods

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#print(os.path.dirname(__file__))
# Llama a un comando de shell desde python
# subprocess.call("python -m spacy download es_core_news_sm", shell=True)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Tokenización
def tokenize_text(text):
    return word_tokenize(text)

# Lematización
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

def lemmatize_text_lang(text, language):
    nlp = spacy.load(f"{language}_core_news_sm")
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# Modificación de la lista de stopwords
def modify_stopwords(text, new_stopwords=[], remove_stopwords=[]):
    stop_words = set(stopwords.words('spanish'))

    # Agrega nuevas stopwords
    for word in new_stopwords:
        stop_words.add(word)

    # Elimina stopwords
    for word in remove_stopwords:
        stop_words.discard(word)

    return ' '.join([word for word in word_tokenize(text) if word not in stop_words])

# Para este primer paso vamos a cargar el corpus 4T
# Define el nombre del archivo
filename = os.path.join(os.path.dirname(__file__), 'Data', 'corpus_4T.tab')
# Lee el archivo CSV
df = pd.read_csv(filename, delimiter='\t', header=None, skiprows=4)
#print(df.head())

# Normalizacion y limpieza
df[1] = df[1].str.lower()

def remove_symbols(text, regex):
    return re.sub(regex, '', text)

df[1] = df[1].apply(lambda x: remove_symbols(x, r'[^\w\s]'))
df[1] = df[1].apply(lambda x: remove_symbols(x, r'\d+'))
print(df.head())

print('El nńumero de elementos antes de eliminar el stopword es: ', len(df[1][0]))
df[1] = df[1].apply(lambda x: modify_stopwords(x, new_stopwords=[], remove_stopwords=[]))
print('El número de elementos antes de Lematizar: ', len(df[1][0]))
df[1] = df[1].apply(lambda x: lemmatize_text_lang(x, 'es'))
print('El número de elementos luego de realizar lematizacion y stopwords', len(df[1][0]))

# Tokenizar
df[1] = df[1].apply(tokenize_text)
print(df[1])

df2 = df.copy()
combinations = CombinationOfMethods(random_selection=2)
combinations.excute_combinations(df2[1], df2[0])
print(combinations.results)
