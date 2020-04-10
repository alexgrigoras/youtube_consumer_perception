#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Process data module
    @alexandru_grigoras
"""

# Libraries
import re
from nltk.corpus import stopwords

# Constants
__all__ = ['ProcessData']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'


class ProcessData:
    """Class for pre-processing the the for the analysis module"""

    def __init__(self):
        """Class constructor"""

        self.__all_tokens = []
        self.__tokens = []

    def process_text(self, text):
        """Process the text by filtering it and removing unwanted characters"""

        self.__tokens.clear()

        tokenize = [t.lower().strip(":,.!?") for t in text.split()]

        filtered_tokens = self.__filter_text(tokenize)

        self.__tokens.extend(filtered_tokens)
        self.__all_tokens.extend(filtered_tokens)

    @staticmethod
    def __filter_text(tokens):
        """Pre-process comments to remove irrelevant data
            Takes in a string of text, then performs the following:
            1. Remove all punctuation
            2. Remove all stopwords
            3. Remove other characters
            4. Return the cleaned text as a list of words"""

        stopwords_english = stopwords.words('english')
        custom_stopwords = []

        hashtags = [w for w in tokens if w.startswith('#')]
        ghashtags = [w for w in tokens if w.startswith('+')]
        mentions = [w for w in tokens if w.startswith('@')]
        links = [w for w in tokens if w.startswith('http') or w.startswith('www')]
        filtered_tokens = [w for w in tokens
                           if w not in stopwords_english
                           and w not in custom_stopwords
                           and w.isalpha()
                           and not len(w) < 3
                           and w not in hashtags
                           and w not in ghashtags
                           and w not in links
                           and w not in mentions]

        return filtered_tokens

    @staticmethod
    def __word_verify(word):
        """Check if the word contains only letters"""

        if re.match("^[a-zA-Z_]*$", word):
            return word.lower()
        else:
            return ''

    def get_tokens(self):
        """Returns the filtered tokens of current process"""

        return self.__tokens

    def get_all_tokens(self):
        """Returns all the filtered tokens"""

        return self.__all_tokens

    def get_word_feature(self, tokens=None):
        """Get the word features from dictionary"""
        
        return dict([(self.__word_verify(word), True) for word in (tokens if tokens else self.__tokens)])
