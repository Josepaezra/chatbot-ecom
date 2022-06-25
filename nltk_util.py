import nltk
#nltk.download('punkt')  # Package with pretrained tokenizer so nltk.word_tokenize works
# from nltk.stem.porter import PorterStemmer  #stem en inglés
import numpy as np
from nltk.stem import SnowballStemmer
# stemmer = PorterStemmer()
stemmer = SnowballStemmer('spanish')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # all_words = [stem(w) for w in all_words]  #Stem para prueba, en train ya se aplica a all_words

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag



"""
#Prueba Tokenización
a = "Cuanto tarda el envío?"
print(a)
a = tokenize(a)
print(a)
"""

"""
#Prueba Stemm
words = ["Habítar", "Conocer", "Planificar"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)
"""

"""
#Prueba Bag of words

oracion = ["cómo", "te", "va", "hola" ]
palabras = ["hol", "buenas", "saludos", "va", "estas"]
bog = bag_of_words(oracion, palabras)
print(bog)
"""
