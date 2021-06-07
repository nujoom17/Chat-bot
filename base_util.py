import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    function for splitting the sentence into an array of individual words
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    example:
    words = ["creation", "creating", "create"]
    words = [stem(w) for w in words]
    o/p = ["creat", "creat", "creat"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    create the bag of words :
    
    example:
    return 1 for each known word that exists in the sentence, otherwise 0
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        if w in sentence_words: 
            bag[i] = 1

    return bag
