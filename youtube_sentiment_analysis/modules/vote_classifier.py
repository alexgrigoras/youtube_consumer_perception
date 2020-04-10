#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Voting system for classifiers
    @alexandru_grigoras
"""

# Libraries
from statistics import mean
from statistics import mode
from nltk.sentiment import SentimentIntensityAnalyzer
from youtube_sentiment_analysis.modules.sentiment_module import sentiment as anew

# Constants
__all__ = ['VoteClassifier']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'


class VoteClassifier:
    """Voting system for classifiers for selecting the most modules sentiment from a list on classifiers"""

    def __init__(self, classifiers):
        """Class constructor"""
        self.__classifiers = classifiers            # ml classifiers
        self.__sid = SentimentIntensityAnalyzer()   # vader classifier

    def classify(self, comment_text, pd):
        """Returns the mean value of the classifiers results"""
        votes = []

        # get ML classifiers results
        for c in self.__classifiers:
            pos = c.prob_classify(pd.get_word_feature()).prob('pos')
            neg = c.prob_classify(pd.get_word_feature()).prob('neg')
            votes.append(float(pos - neg))

        # get Vader result
        ss = self.__sid.polarity_scores(comment_text)
        votes.append(ss["compound"])

        # get ANEW result
        anew_result = anew.sentiment(pd.get_tokens())['valence']
        votes.append(self.map(anew_result, 0, 10, -1, 1))

        return mean(votes)

    def confidence(self, comment_text, pd):
        """Returns the confidence of the result"""
        votes = []

        # get ML classifiers result
        for c in self.__classifiers:
            v = c.classify(pd.get_word_feature())
            votes.append(v)

        # get Vader result
        ss = self.__sid.polarity_scores(comment_text)
        if ss["compound"] >= 0:
            votes.append("pos")
        else:
            votes.append("neg")

        # get ANEW result
        anew_result = anew.sentiment(pd.get_tokens())['valence']
        if anew_result >= 5.8:
            votes.append("pos")
        else:
            votes.append("neg")

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / float(len(votes))

        return conf

    @staticmethod
    def map(value, left_min, left_max, right_min, right_max):
        """Maps a value from one interval [left_min, left_max] to another [right_min, right_max]"""
        # Figure out how 'wide' each range is
        left_span = left_max - left_min
        right_span = right_max - right_min

        # Convert the left range into a 0-1 range (float)
        value_scaled = float(value - left_min) / float(left_span)

        # Convert the 0-1 range into a value in the right range.
        return right_min + (value_scaled * right_span)

