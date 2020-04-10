#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Classifiers accuracy
    @alexandru_grigoras
"""

# Libraries
import time

import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVC

from youtube_sentiment_analysis.modules.process import ProcessData
from youtube_sentiment_analysis.modules.sentiment_module import sentiment as anew
from youtube_sentiment_analysis.modules.training import TrainClassifier
from youtube_sentiment_analysis.modules.vote_classifier import VoteClassifier

# Constants

__all__ = ['TestAccuracy']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'

CLASSIFIERS_PATH = "youtube_sentiment_analysis/data/classifiers_large_dataset/"


class TestAccuracy:
    """Class for testing the accuracy of the algorithms"""

    def __init__(self, dataset_path, max_nr_docs):
        """Class constructor"""
        self.__dataset_path = dataset_path
        self.__max_nr_docs = max_nr_docs

    def test_cross_val_score(self, k_fold, progress, console):
        """Testing Classifiers Accuracy using Cross-Validation Method"""

        # get the starting time
        start_time = time.time()

        tc = TrainClassifier(self.__dataset_path, self.__max_nr_docs)

        text_data, label_values = tc.get_dataset_labeled()

        x_elements = np.array(text_data)
        y_elements = np.array(label_values)

        sid = SentimentIntensityAnalyzer()

        vader_accuracy = []
        anew_accuracy = []
        voting_accuracy = []

        process_data = ProcessData()

        progress_value = 0

        # machine learning classifiers
        classifiers = tc.get_classifiers(progress, console)
        voted_classifier = VoteClassifier(classifiers)

        kf = KFold(n_splits=k_fold, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(x_elements):
            x_train, x_test = x_elements[train_index], x_elements[test_index]
            y_train, y_test = y_elements[train_index], y_elements[test_index]

            test_values_vader = []
            test_values_anew = []
            test_values_voting = []
            predicted_values = []

            for text, value in zip(x_test, y_test):
                process_data.process_text(text)

                ss = sid.polarity_scores(text)

                if ss["compound"] >= 0:
                    test_values_vader.append("positive")
                else:
                    test_values_vader.append("negative")

                tokens = process_data.get_tokens()

                if anew.sentiment(tokens)['valence'] >= 5.8:
                    test_values_anew.append("positive")
                else:
                    test_values_anew.append("negative")

                if value == -1:
                    predicted_values.append("negative")
                else:
                    predicted_values.append("positive")

                # machine learning algorithms sentiment value
                ml_algorithms_sentiment = voted_classifier.classify(text, process_data)

                if ml_algorithms_sentiment >= 0:
                    test_values_voting.append("positive")
                else:
                    test_values_voting.append("negative")

            acc_vader = accuracy_score(test_values_vader, predicted_values, normalize=True)
            acc_anew = accuracy_score(test_values_anew, predicted_values, normalize=True)
            acc_voting = accuracy_score(test_values_voting, predicted_values, normalize=True)

            vader_accuracy.append(acc_vader)
            anew_accuracy.append(acc_anew)
            voting_accuracy.append(acc_voting)

            progress_value += 40 / k_fold
            progress.setValue(progress_value)

        vader_accuracy_array = np.array(vader_accuracy)
        anew_accuracy_array = np.array(anew_accuracy)
        voting_accuracy_array = np.array(voting_accuracy)

        console.append("> %s: %f (%f)" % ("VADER", vader_accuracy_array.mean(), vader_accuracy_array.std()))
        console.append("> %s: %f (%f)" % ("ANEW", anew_accuracy_array.mean(), anew_accuracy_array.std()))
        console.append("> %s: %f (%f)" % ("VOTING", voting_accuracy_array.mean(), voting_accuracy_array.std()))

        # prepare configuration for cross validation test harness
        models = [('NuSVC', NuSVC(nu=0.5, kernel='linear', probability=True, gamma='scale', cache_size=500,
                                  class_weight='balanced')),
                  ('LR', LogisticRegression(penalty='l2', class_weight='balanced', solver='saga', max_iter=5000,
                                            n_jobs=-1, warm_start=True)),
                  ('MNB', MultinomialNB(alpha=1))]

        # evaluate each model in turn
        results = []
        names = []
        show_info = 0

        # add the VADER and ANEW classifiers
        results.append(voting_accuracy)
        names.append("VOTING")

        for name, model in models:
            tf_idf = TfidfVectorizer()
            classifier = make_pipeline(tf_idf, model)
            cv_results = model_selection.cross_val_score(classifier, x_elements, y_elements,
                                                         cv=k_fold, scoring='accuracy', n_jobs=-1,
                                                         verbose=show_info)
            results.append(cv_results)
            names.append(name)
            console.append("> %s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

            progress_value += 20
            progress.setValue(progress_value)

        # add vader and anew classifiers
        results.append(vader_accuracy)
        names.append("VADER")
        results.append(anew_accuracy)
        names.append("ANEW")

        # get the ending time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        console.append("> Data processed in " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + " seconds")

        return results, names

