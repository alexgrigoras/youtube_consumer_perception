#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Classifiers training
    @alexandru_grigoras
"""

# Libraries
import glob
import os
import pickle
import time

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, NuSVC

from youtube_sentiment_analysis.modules.process import ProcessData

# Constants
__all__ = ['TrainClassifier']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'

CLASSIFIERS_PATH = "youtube_sentiment_analysis/data/classifiers/"


class TrainClassifier:
    """Training class for training and saving the classifiers"""

    def __init__(self, dataset_path=None, max_nr_docs=None, classifiers_names=None):
        """Class constructor"""
        self.__classifiers_names = []
        self.__trained_classifiers = []

        if not classifiers_names or classifiers_names == [""]:
            self.__classifiers_names = ['multinomial_naive_bayes', 'logistic_regression', 'nu_svc']
        else:
            for classifier in classifiers_names:
                self.__classifiers_names.append(classifier)

        self.__dataset_path = dataset_path
        self.__max_nr_docs = max_nr_docs

    def set_classifiers(self, classifiers_names):
        """Sets the classifiers to be trained"""

        self.__classifiers_names.clear()
        for classifier in classifiers_names:
            self.__classifiers_names.append(classifier)

    def train(self, progress, console):
        """Train the classifiers with the data from nltk library"""

        console.append("> Selected classifiers: " + str(self.__classifiers_names))

        progress_value = 0
        nr_classifiers = len(self.__classifiers_names)

        for classifier_name in self.__classifiers_names:
            console.append("> Training the classifier: " + classifier_name)
            # get the starting time
            classifier_start_time = time.time()

            train_set, test_set = self.get_dataset_split()

            if classifier_name == 'multinomial_naive_bayes':
                cl_name = "Multinomial NB classifier"
                classifier = SklearnClassifier(MultinomialNB(alpha=1))

            elif classifier_name == 'bernoulli_naive_bayes':
                cl_name = "Bernoulli NB Classifier"
                classifier = SklearnClassifier(BernoulliNB(alpha=1, binarize=0))

            elif classifier_name == 'logistic_regression':
                cl_name = "Logistic Regression"
                classifier = SklearnClassifier(LogisticRegression(penalty='l2', class_weight='balanced', solver='saga',
                                                                  max_iter=2000, n_jobs=-1, warm_start=True))

            elif classifier_name == 'svc':
                cl_name = "SVC"
                classifier = SklearnClassifier(SVC(kernel='linear', probability=True, gamma='scale',
                                                   class_weight='balanced', max_iter=2000, cache_size=300))

            elif classifier_name == 'nu_svc':
                cl_name = "Nu SVC"
                classifier = SklearnClassifier(NuSVC(nu=0.5, kernel='linear', probability=True, gamma='scale',
                                                     max_iter=2000, cache_size=300, class_weight='balanced'))

            else:
                console.append("> Invalid classifier name")
                return

            classifier.train(train_set)
            console.append("> " + cl_name + " accuracy percent: " +
                           str((nltk.classify.accuracy(classifier, test_set)) * 100) + "%")

            self.__save_classifier(classifier_name, classifier)
            self.__trained_classifiers.append(classifier)

            progress_value += 100/nr_classifiers
            progress.setValue(progress_value)

            # get the ending time and calculate elapsed time
            classifier_end_time = time.time()
            classifier_elapsed_time = classifier_end_time - classifier_start_time
            console.append("> Training " + classifier_name + " finished in " +
                           time.strftime("%H:%M:%S", time.gmtime(classifier_elapsed_time)) + " seconds")

    def get_classifiers(self, progress=None, console=None):
        """Returns the trained classifiers or trains them"""

        classifiers = []

        read_directory = os.listdir(CLASSIFIERS_PATH)

        if len(read_directory) == 0:
            console.append("> Training the classifiers: ")
            self.train(progress, console)
            classifiers = self.get_trained_classifiers()
        else:
            console.append("> Getting the trained classifiers: ")
            file_nr = 1
            for f in read_directory:
                console.append("  " + str(file_nr) + ". " + f)
                file_nr = file_nr + 1
                classifiers.append(self.open_classifier(f))
            console.append("  " + str(file_nr) + ". vader classifier")
            console.append("  " + str(file_nr + 1) + ". anew classifier")

        return classifiers

    def get_trained_classifiers(self):
        """Returns a list with trained classifiers objects"""

        return self.__trained_classifiers

    @staticmethod
    def __save_classifier(_name, _classifier):
        """Save in file to avoid training the data again"""

        save_document = open(CLASSIFIERS_PATH + _name + ".pickle", 'wb')
        pickle.dump(_classifier, save_document)
        save_document.close()

    @staticmethod
    def open_classifier(name):
        """Open the trained classifier with the data from nltk library"""

        open_file = open(CLASSIFIERS_PATH + name, 'rb')
        classifier = pickle.load(open_file, encoding='bytes')
        open_file.close()

        return classifier

    def get_dataset_split(self):
        """Get dataset from files (negative and positive words)
            25000 train + 25000 test (imdb)"""

        file_path_train_neg = glob.glob(self.__dataset_path + 'train/neg/*.txt')
        file_path_test_neg = glob.glob(self.__dataset_path + 'test/neg/*.txt')
        file_path_train_pos = glob.glob(self.__dataset_path + 'train/pos/*.txt')
        file_path_test_pos = glob.glob(self.__dataset_path + 'test/pos/*.txt')

        neg_train_ids = []
        pos_train_ids = []
        neg_test_ids = []
        pos_test_ids = []

        pd = ProcessData()

        max_docs = self.__max_nr_docs / 4

        # train data
        nr_docs = 0
        for fp in file_path_train_neg:
            with open(fp, 'r') as f:
                if nr_docs < max_docs or self.__max_nr_docs is -1:
                    neg_train_ids = neg_train_ids + [(pd.get_word_feature(f.read().split()), 'neg')]
                nr_docs = nr_docs + 1
        nr_docs = 0
        for fp in file_path_train_pos:
            with open(fp, 'r') as f:
                if nr_docs < max_docs or self.__max_nr_docs is -1:
                    pos_train_ids = pos_train_ids + [(pd.get_word_feature(f.read().split()), 'pos')]
                nr_docs = nr_docs + 1

        # test data
        nr_docs = 0
        for fp in file_path_test_neg:
            with open(fp, 'r') as f:
                if nr_docs < max_docs or self.__max_nr_docs is -1:
                    neg_test_ids = neg_test_ids + [(pd.get_word_feature(f.read().split()), 'neg')]
                nr_docs = nr_docs + 1
        nr_docs = 0
        for fp in file_path_test_pos:
            with open(fp, 'r') as f:
                if nr_docs < max_docs / 4 or self.__max_nr_docs is -1:
                    pos_test_ids = pos_test_ids + [(pd.get_word_feature(f.read().split()), 'pos')]
                nr_docs = nr_docs + 1

        # concatenate data
        train_set = neg_train_ids + pos_train_ids
        test_set = neg_test_ids + pos_test_ids

        return train_set, test_set

    def get_dataset_labeled(self):
        """Get dataset from files (negative and positive words)
            25000 train + 25000 test (imdb) with labels"""

        # files path
        file_path_neg = glob.glob(self.__dataset_path + 'train/neg/*.txt') + \
                        glob.glob(self.__dataset_path + 'test/neg/*.txt')
        file_path_pos = glob.glob(self.__dataset_path + 'train/pos/*.txt') + \
                        glob.glob(self.__dataset_path + 'test/pos/*.txt')

        text_data = []
        label_values = []

        max_docs = self.__max_nr_docs / 2

        # negative comments
        nr_docs = 0
        for fp in file_path_neg:
            with open(fp, 'r') as f:
                if nr_docs < max_docs or self.__max_nr_docs is -1:
                    text_data.append(f.read())
                    label_values.append(-1)
                nr_docs = nr_docs + 1

        # positive comments
        nr_docs = 0
        for fp in file_path_pos:
            with open(fp, 'r') as f:
                if nr_docs < max_docs or self.__max_nr_docs is -1:
                    text_data.append(f.read())
                    label_values.append(1)
                nr_docs = nr_docs + 1

        return text_data, label_values
