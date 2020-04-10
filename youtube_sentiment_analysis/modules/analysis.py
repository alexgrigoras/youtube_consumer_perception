#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Data analysis module
    @alexandru_grigoras
"""

# Libraries
from __future__ import print_function

import time

from nltk.probability import *

from youtube_sentiment_analysis.modules.process import ProcessData
from youtube_sentiment_analysis.modules.sentiment_module import sentiment as anew
from youtube_sentiment_analysis.modules.store import StoreData
from youtube_sentiment_analysis.modules.training import TrainClassifier
from youtube_sentiment_analysis.modules.vote_classifier import VoteClassifier

# Constants
__all__ = ['DataAnalysis']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'


class DataAnalysis:
    """Analyse the data and determine sentiment and word frequency"""

    def __init__(self, keyword, like_threshold_min, like_threshold_max):
        self.__keyword = keyword
        self.__like_threshold_min = like_threshold_min
        self.__like_threshold_max = like_threshold_max

    def get_data_from_DB(self):
        """Get the downloaded videos data from MongoDB using store module"""
        # create a MongoDB connection
        mongo_conn = StoreData(self.__keyword, store=False)

        # get the data from MongoDB
        videos_data = mongo_conn.read()

        return videos_data

    def analyse(self, progress, console):
        """Analyse data and prepare it for display module"""
        # get the starting time
        start_time = time.time()

        # variables
        videos = []
        likes = []
        author = []
        comm_time = []
        comments = []
        sentiment_val = []
        confidence_val = []
        sentiment_anew_arousal = []

        # get machine learning classifiers
        tc = TrainClassifier()
        classifiers = tc.get_classifiers(progress, console)

        # vote classifier object
        voted_classifier = VoteClassifier(classifiers)

        # process data object
        pd = ProcessData()

        # get data
        videos_data = self.get_data_from_DB()

        nr_videos = videos_data.count()

        progress_value = 0

        # parse data
        for video in videos_data:

            get_comments = video.get("comments")
            nr_comments = len(get_comments)

            for comment in get_comments:
                # get likes
                like = float(comment.get("nr_likes"))

                if self.__like_threshold_min <= like <= self.__like_threshold_max:
                    videos.append(video.get("title"))
                    likes.append(like)
                    author.append(comment.get("author"))
                    comm_time.append(comment.get("time"))

                    # get comments and apply filters
                    comment_text = comment.get("text")
                    comments.append(comment_text)
                    pd.process_text(comment_text)

                    # machine learning algorithms sentiment value with voting system
                    ml_algorithms_sentiment = voted_classifier.classify(comment_text, pd)
                    sentiment_val.append(ml_algorithms_sentiment)

                    # machine learning algorithms confidence value with voting system
                    ml_algorithms_confidence = voted_classifier.confidence(comment_text, pd)
                    confidence_val.append(ml_algorithms_confidence)

                    # get ANEW arousal values
                    anew_result_arousal = anew.sentiment(pd.get_tokens())['arousal']
                    sentiment_anew_arousal.append(anew_result_arousal)

                progress_value += 80 / nr_videos / nr_comments
                progress.setValue(progress_value)

        if not pd.get_all_tokens():
            return

        # FreqDist returns a list of tuples containing each word and the number of its occurences
        fd = FreqDist(pd.get_all_tokens())

        # get the ending time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        console.append("> Data processed in " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + " seconds")

        return fd, pd, sentiment_val, sentiment_anew_arousal, likes, confidence_val, comments, videos, author, comm_time


