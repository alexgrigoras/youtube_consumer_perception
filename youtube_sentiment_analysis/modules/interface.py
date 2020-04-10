#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    User interface
    @alexandru_grigoras
"""

# Libraries
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QAbstractItemView
from pymongo.errors import PyMongoError

from youtube_sentiment_analysis.modules.accuracy import TestAccuracy
from youtube_sentiment_analysis.modules.analysis import DataAnalysis
from youtube_sentiment_analysis.modules.crawler import WebCrawler
from youtube_sentiment_analysis.modules.display import DisplayData
from youtube_sentiment_analysis.modules.store import StoreData
from youtube_sentiment_analysis.modules.training import TrainClassifier

# Constants
__all__ = ['SentimentAnalysisApplication']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'


class UIMainWindow(object):
    """Main User Interface"""

    def setupUi(self, MainWindow):
        """Setup the objects for ui"""

        # main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1366, 768)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)

        #central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidgetData = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidgetData.setGeometry(QtCore.QRect(390, 20, 951, 691))
        self.tabWidgetData.setObjectName("tabWidgetData")

        # sentiment page
        self.sentiment = QtWidgets.QWidget()
        self.sentiment.setObjectName("sentiment")
        self.tabWidgetData.addTab(self.sentiment, "")

        # heatmap page
        self.heatmap = QtWidgets.QWidget()
        self.heatmap.setObjectName("heatmap")
        self.tabWidgetData.addTab(self.heatmap, "")

        # word frequency page
        self.word_frequency = QtWidgets.QWidget()
        self.word_frequency.setObjectName("word_frequency")
        self.tabWidgetData.addTab(self.word_frequency, "")

        # word cloud page
        self.word_cloud = QtWidgets.QWidget()
        self.word_cloud.setObjectName("word_cloud")
        self.tabWidgetData.addTab(self.word_cloud, "")

        # comments page
        self.comments = QtWidgets.QWidget()
        self.comments.setObjectName("comments")
        self.tabWidgetData.addTab(self.comments, "")
        self.treeView = QtWidgets.QTreeView(self.comments)
        self.treeView.setGeometry(QtCore.QRect(30, 20, 891, 611))
        self.treeView.setObjectName("treeView")

        # accuracy page
        self.accuracy = QtWidgets.QWidget()
        self.accuracy.setObjectName("accuracy")
        self.tabWidgetData.addTab(self.accuracy, "")

        # settings page
        self.settings = QtWidgets.QWidget()
        self.settings.setObjectName("settings")
        self.groupBoxComments = QtWidgets.QGroupBox(self.settings)
        self.groupBoxComments.setGeometry(QtCore.QRect(30, 50, 331, 151))
        self.groupBoxComments.setObjectName("groupBoxComments")
        self.labelLikeMin = QtWidgets.QLabel(self.groupBoxComments)
        self.labelLikeMin.setGeometry(QtCore.QRect(30, 50, 171, 31))
        self.labelLikeMin.setObjectName("labelLikeMin")
        self.labelLikeMax = QtWidgets.QLabel(self.groupBoxComments)
        self.labelLikeMax.setGeometry(QtCore.QRect(30, 90, 171, 31))
        self.labelLikeMax.setObjectName("labelLikeMax")
        self.lineEditLikeMin = QtWidgets.QLineEdit(self.groupBoxComments)
        self.lineEditLikeMin.setGeometry(QtCore.QRect(230, 50, 81, 28))
        self.lineEditLikeMin.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.lineEditLikeMin.setObjectName("lineEditLikeMin")
        self.lineEditLikeMax = QtWidgets.QLineEdit(self.groupBoxComments)
        self.lineEditLikeMax.setGeometry(QtCore.QRect(230, 90, 81, 28))
        self.lineEditLikeMax.setObjectName("lineEditLikeMax")
        self.groupBoxTraining = QtWidgets.QGroupBox(self.settings)
        self.groupBoxTraining.setGeometry(QtCore.QRect(30, 240, 601, 321))
        self.groupBoxTraining.setObjectName("groupBoxTraining")
        self.labelDataset = QtWidgets.QLabel(self.groupBoxTraining)
        self.labelDataset.setGeometry(QtCore.QRect(30, 50, 111, 31))
        self.labelDataset.setObjectName("labelDataset")
        self.lineEditDataset = QtWidgets.QLineEdit(self.groupBoxTraining)
        self.lineEditDataset.setGeometry(QtCore.QRect(150, 50, 421, 28))
        self.lineEditDataset.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.lineEditDataset.setText("")
        self.lineEditDataset.setObjectName("lineEditDataset")
        self.pushButtonTrain = QtWidgets.QPushButton(self.groupBoxTraining)
        self.pushButtonTrain.setGeometry(QtCore.QRect(30, 160, 101, 31))
        self.pushButtonTrain.setObjectName("pushButtonTrain")
        self.pushButtonTrain.clicked.connect(self.__on_click_train)
        self.labelDatasetLimit = QtWidgets.QLabel(self.groupBoxTraining)
        self.labelDatasetLimit.setGeometry(QtCore.QRect(30, 100, 111, 31))
        self.labelDatasetLimit.setObjectName("labelDatasetLimit")
        self.lineEditDatasetLimit = QtWidgets.QLineEdit(self.groupBoxTraining)
        self.lineEditDatasetLimit.setGeometry(QtCore.QRect(150, 100, 131, 28))
        self.lineEditDatasetLimit.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.lineEditDatasetLimit.setText("200")
        self.lineEditDatasetLimit.setObjectName("lineEditDatasetLimit")
        self.lineEditDatasetKfold = QtWidgets.QLineEdit(self.groupBoxTraining)
        self.lineEditDatasetKfold.setGeometry(QtCore.QRect(150, 210, 131, 28))
        self.lineEditDatasetKfold.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.lineEditDatasetKfold.setText("10")
        self.lineEditDatasetKfold.setObjectName("lineEditDatasetKfold")
        self.labelDatasetKfold = QtWidgets.QLabel(self.groupBoxTraining)
        self.labelDatasetKfold.setGeometry(QtCore.QRect(30, 210, 111, 31))
        self.labelDatasetKfold.setObjectName("labelDatasetKfold")
        self.pushButtonAccuracy = QtWidgets.QPushButton(self.groupBoxTraining)
        self.pushButtonAccuracy.setGeometry(QtCore.QRect(30, 260, 101, 31))
        self.pushButtonAccuracy.setObjectName("pushButtonAccuracy")
        self.pushButtonAccuracy.clicked.connect(self.__on_click_accuracy)
        self.tabWidgetData.addTab(self.settings, "")

        # group box search
        self.groupBoxSearch = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxSearch.setGeometry(QtCore.QRect(20, 20, 351, 101))
        self.groupBoxSearch.setObjectName("groupBoxSearch")
        self.lineEditSearch = QtWidgets.QLineEdit(self.groupBoxSearch)
        self.lineEditSearch.setGeometry(QtCore.QRect(20, 40, 311, 41))
        self.lineEditSearch.setObjectName("lineEditSearch")
        self.groupBoxAnalyse = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxAnalyse.setGeometry(QtCore.QRect(20, 240, 351, 151))
        self.groupBoxAnalyse.setObjectName("groupBoxAnalyse")

        # group box data
        self.groupBoxData = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxData.setGeometry(QtCore.QRect(20, 130, 351, 101))
        self.groupBoxData.setObjectName("groupBoxData")
        self.comboBoxDB = QtWidgets.QComboBox(self.groupBoxData)
        self.comboBoxDB.setGeometry(QtCore.QRect(70, 40, 261, 41))
        self.comboBoxDB.setAcceptDrops(False)
        self.comboBoxDB.setObjectName("comboBoxDB")
        self.pushButtonLoad = QtWidgets.QPushButton(self.groupBoxData)
        self.pushButtonLoad.setGeometry(QtCore.QRect(20, 40, 41, 41))
        self.pushButtonLoad.setObjectName("pushButtonLoad")
        self.pushButtonLoad.clicked.connect(self.__on_click_load)

        # button extract
        self.pushButtonExtract = QtWidgets.QPushButton(self.groupBoxAnalyse)
        self.pushButtonExtract.setGeometry(QtCore.QRect(130, 40, 91, 61))
        self.pushButtonExtract.setObjectName("pushButtonExtract")
        self.pushButtonExtract.clicked.connect(self.__on_click_extract)

        # button analyse
        self.pushButtonAnalyse = QtWidgets.QPushButton(self.groupBoxAnalyse)
        self.pushButtonAnalyse.setGeometry(QtCore.QRect(240, 40, 91, 61))
        self.pushButtonAnalyse.setObjectName("pushButtonAnalyse")
        self.pushButtonAnalyse.clicked.connect(self.__on_click_analyse)
        self.pushButtonAnalyse.setEnabled(False)

        # number of videos
        self.lineEditNrVideos = QtWidgets.QLineEdit(self.groupBoxAnalyse)
        self.lineEditNrVideos.setGeometry(QtCore.QRect(20, 70, 91, 28))
        self.lineEditNrVideos.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.lineEditNrVideos.setText("")
        self.lineEditNrVideos.setObjectName("lineEditNrVideos")
        self.labelNrVideos1 = QtWidgets.QLabel(self.groupBoxAnalyse)
        self.labelNrVideos1.setGeometry(QtCore.QRect(30, 30, 71, 21))
        self.labelNrVideos1.setObjectName("labelNrVideos1")
        self.labelNrVideos2 = QtWidgets.QLabel(self.groupBoxAnalyse)
        self.labelNrVideos2.setGeometry(QtCore.QRect(40, 50, 51, 21))
        self.labelNrVideos2.setObjectName("labelNrVideos2")

        # progress bar
        self.progressBar = QtWidgets.QProgressBar(self.groupBoxAnalyse)
        self.progressBar.setGeometry(QtCore.QRect(20, 110, 311, 21))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        # console
        self.groupBoxConsole = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxConsole.setGeometry(QtCore.QRect(20, 410, 351, 301))
        self.groupBoxConsole.setObjectName("groupBoxConsole")
        self.textEditConsole = QtWidgets.QTextEdit(self.groupBoxConsole)
        self.textEditConsole.setGeometry(QtCore.QRect(20, 40, 311, 241))
        self.textEditConsole.setObjectName("textEditConsole")

        MainWindow.setCentralWidget(self.centralwidget)

        # menu bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1366, 25))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.setStatusTip('Leave The App')
        self.actionExit.triggered.connect(self.__close_application)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout.setStatusTip('Informations about the app')
        self.actionAbout.triggered.connect(self.__about_application)
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        # display data module
        width_val = 9.5
        height_val = 6.5
        self.sentiment_display = DisplayData(self.sentiment, width=width_val, height=height_val)
        self.heatmap_display = DisplayData(self.heatmap, width=width_val, height=height_val)
        self.word_frequency_display = DisplayData(self.word_frequency, width=width_val, height=height_val)
        self.word_cloud_display = DisplayData(self.word_cloud, width=width_val, height=height_val)
        self.accuracy_display = DisplayData(self.accuracy, width=width_val, height=height_val)

        self.__retranslateUi(MainWindow)
        self.tabWidgetData.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def __retranslateUi(self, MainWindow):
        """Sets the label names and other paramenters"""

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YouTube Sentiment Analysis"))

        self.tabWidgetData.setTabText(self.tabWidgetData.indexOf(self.sentiment), _translate("MainWindow", "Sentiment"))
        self.tabWidgetData.setTabText(self.tabWidgetData.indexOf(self.heatmap), _translate("MainWindow", "Heatmap"))
        self.tabWidgetData.setTabText(self.tabWidgetData.indexOf(self.word_frequency),
                                      _translate("MainWindow", "Word Frequency"))
        self.tabWidgetData.setTabText(self.tabWidgetData.indexOf(self.word_cloud),
                                      _translate("MainWindow", "WordCloud"))
        self.tabWidgetData.setTabText(self.tabWidgetData.indexOf(self.comments), _translate("MainWindow", "Comments"))
        self.tabWidgetData.setTabText(self.tabWidgetData.indexOf(self.accuracy), _translate("MainWindow", "Accuracy"))
        self.groupBoxComments.setTitle(_translate("MainWindow", "Comments"))
        self.labelLikeMin.setText(_translate("MainWindow", "Minimum number of likes:"))
        self.labelLikeMax.setText(_translate("MainWindow", "Maximum number of likes:"))
        self.lineEditLikeMin.setText(_translate("MainWindow", "10"))
        self.lineEditLikeMax.setText(_translate("MainWindow", "1000"))
        self.lineEditDataset.setText(_translate("MainWindow", "/home/alex/imdb_data/"))
        self.groupBoxTraining.setTitle(_translate("MainWindow", "Training and Accuracy"))
        self.labelDataset.setText(_translate("MainWindow", "Dataset path:"))
        self.pushButtonTrain.setText(_translate("MainWindow", "Train"))
        self.labelDatasetLimit.setText(_translate("MainWindow", "Limit documents:"))
        self.labelDatasetKfold.setText(_translate("MainWindow", "k-folds:"))
        self.pushButtonAccuracy.setText(_translate("MainWindow", "Accuracy"))
        self.groupBoxSearch.setTitle(_translate("MainWindow", "Search Data"))
        self.groupBoxAnalyse.setTitle(_translate("MainWindow", "Analyse Data"))
        self.pushButtonExtract.setText(_translate("MainWindow", "Extract"))
        self.pushButtonAnalyse.setText(_translate("MainWindow", "Analyse"))
        self.pushButtonAccuracy.setText(_translate("MainWindow", "Acccuracy"))
        self.groupBoxConsole.setTitle(_translate("MainWindow", "Console"))
        self.tabWidgetData.setTabText(self.tabWidgetData.indexOf(self.settings), _translate("MainWindow", "Settings"))
        self.labelNrVideos1.setText(_translate("MainWindow", "Number of"))
        self.labelNrVideos2.setText(_translate("MainWindow", "videos:"))
        self.pushButtonLoad.setText(_translate("MainWindow", "Load"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.groupBoxData.setTitle(_translate("MainWindow", "Data from database"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(184, 207, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(184, 207, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.pushButtonAnalyse.setPalette(palette)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(184, 207, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(184, 207, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        self.comboBoxDB.setPalette(palette)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(184, 207, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(184, 207, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(196, 225, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(155, 192, 231))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(76, 106, 138))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(57, 79, 103))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(114, 159, 207))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.pushButtonLoad.setPalette(palette)

    @staticmethod
    def __about_application():
        """Open a popup with application details"""

        message_box = QMessageBox()
        message_box.about(message_box, "About", "Youtube Sentiment Analysis Application\n"
                                                "Degree project\n"
                                                "Author: Alexandru Grigoras")

    @staticmethod
    def __close_application():
        """Close the application"""

        sys.exit()

    def __on_click_load(self):
        """Load the collection names"""

        # get data
        sd = StoreData()
        try:
            collections = sd.get_collections()

            collections.sort()

            self.comboBoxDB.clear()
            self.comboBoxDB.addItems(collections)

            self.pushButtonAnalyse.setEnabled(True)
        except PyMongoError:
            self.textEditConsole.append("> Database server is not opened!")

    @pyqtSlot(name="extract")
    def __on_click_extract(self):
        """Extracts the data from YouTube"""

        input_text = self.lineEditSearch.text()

        if input_text is not None:
            if input_text is not "":
                # get like threshold
                try:
                    like_threshold_min = int(self.lineEditLikeMin.text())

                    if like_threshold_min < 0:
                        self.textEditConsole.append("> The minimum number of likes should be positive")
                        return

                except ValueError:
                    self.textEditConsole.append("> The minimum number of likes is not valid")
                    return

                try:
                    like_threshold_max = int(self.lineEditLikeMax.text())

                    if like_threshold_max < 0:
                        self.textEditConsole.append("> The maximum number of likes should be positive")
                        return

                except ValueError:
                    self.textEditConsole.append("> The maximum number of likes is not valid")
                    return

                try:
                    nr_videos = int(self.lineEditNrVideos.text())

                    if nr_videos < 0:
                        self.textEditConsole.append("> The number of videos should be positive")
                        return

                except ValueError:
                    self.textEditConsole.append("> The number of videos is not valid")
                    return

                # extract data
                crawl_delay = 1
                extracted_data = WebCrawler(input_text, nr_videos, crawl_delay)
                crawling_result = extracted_data.run(self.textEditConsole)

                if crawling_result is True:
                    # process data
                    data = DataAnalysis(input_text, like_threshold_min, like_threshold_max)
                    try:
                        fd, pd, sentiment_val, sentiment_anew_arousal, likes, confidence, comments, videos, author, \
                            comm_time = data.analyse(self.progressBar, self.textEditConsole)
                    except TypeError:
                        self.textEditConsole.append("> No data found or like threshold is too big!")
                        return

                    # clear any plot if it exists
                    self.sentiment_display.clear_plot()
                    self.heatmap_display.clear_plot()
                    self.word_frequency_display.clear_plot()
                    self.word_cloud_display.clear_plot()

                    progress_value = 80

                    # plot data
                    self.sentiment_display.plot_classifiers(sentiment_val, sentiment_anew_arousal, likes, confidence,
                                                            'Sentiment', 'Activare', comments, videos, author,
                                                            comm_time)
                    progress_value += 5
                    self.progressBar.setValue(progress_value)
                    self.heatmap_display.plot_heatmap(sentiment_val, sentiment_anew_arousal, "Sentiment", "Activare")
                    progress_value += 5
                    self.progressBar.setValue(progress_value)
                    self.word_frequency_display.plot_word_frequency(fd.items())
                    progress_value += 5
                    self.progressBar.setValue(progress_value)
                    self.word_cloud_display.plot_wordcloud(pd.get_all_tokens())
                    progress_value += 5
                    self.progressBar.setValue(progress_value)

                    # put comments to treeview
                    self.__populate_treeview(data)
            else:
                self.textEditConsole.append("> The input text is empty!")
        else:
            self.textEditConsole.append("> Invalid input text!")

    @pyqtSlot(name="analyse")
    def __on_click_analyse(self):
        """Analyses Data"""

        input_text = self.comboBoxDB.currentText()

        if input_text is not None:
            if input_text is not "":

                try:
                    like_threshold_min = int(self.lineEditLikeMin.text())

                    if like_threshold_min < 0:
                        self.textEditConsole.append("> The minimum number of likes should be positive")
                        return

                except ValueError:
                    self.textEditConsole.append("> The minimum number of likes is not valid")
                    return

                try:
                    like_threshold_max = int(self.lineEditLikeMax.text())

                    if like_threshold_max < 0:
                        self.textEditConsole.append("> The maximum number of likes should be positive")
                        return

                    if like_threshold_max <= like_threshold_min:
                        self.textEditConsole.append("> The maximum number of likes should greater than "
                                                    "the minimum number of likes")
                        return

                except ValueError:
                    self.textEditConsole.append("> The maximum number of likes is not valid")
                    return

                # process data
                data = DataAnalysis(input_text, like_threshold_min, like_threshold_max)
                try:
                    fd, pd, sentiment_val, sentiment_anew_arousal, likes, confidence, comments, videos, author, \
                        comm_time = data.analyse(self.progressBar, self.textEditConsole)
                except TypeError:
                    self.textEditConsole.append("> No data found or like threshold is too large")
                    return

                # clear any plot if it exists
                self.sentiment_display.clear_plot()
                self.heatmap_display.clear_plot()
                self.word_frequency_display.clear_plot()
                self.word_cloud_display.clear_plot()

                progress_value = 80

                # plot data
                self.sentiment_display.plot_classifiers(sentiment_val, sentiment_anew_arousal, likes, confidence,
                                                        'Sentiment', 'Activare', comments, videos, author, comm_time)
                progress_value += 5
                self.progressBar.setValue(progress_value)
                self.heatmap_display.plot_heatmap(sentiment_val, sentiment_anew_arousal, "Sentiment", "Activare")
                progress_value += 5
                self.progressBar.setValue(progress_value)
                self.word_frequency_display.plot_word_frequency(fd.items())
                progress_value += 5
                self.progressBar.setValue(progress_value)
                self.word_cloud_display.plot_wordcloud(pd.get_all_tokens())
                progress_value += 5
                self.progressBar.setValue(progress_value)

                # put comments to treeview
                self.__populate_treeview(data)
            else:
                self.textEditConsole.append("> The input text is empty!")
        else:
            self.textEditConsole.append("> Invalid input text!")

    @pyqtSlot(name="train")
    def __on_click_train(self):
        """Train the classifiers"""

        try:
            max_nr_docs = int(self.lineEditDatasetLimit.text())

            if max_nr_docs < 1:
                self.textEditConsole.append("> The maximum number of documents should be positive")
                return

        except ValueError:
            self.textEditConsole.append("> The maximum number of documents is not valid")
            return

        dataset_path = self.lineEditDataset.text()

        train_classifier = TrainClassifier(dataset_path, max_nr_docs)
        train_classifier.train(self.progressBar, self.textEditConsole)

    @pyqtSlot(name="accuracy")
    def __on_click_accuracy(self):
        """Test the accuracy of the classifiers"""

        try:
            max_nr_docs = int(self.lineEditDatasetLimit.text())

            if max_nr_docs < 1:
                self.textEditConsole.append("> The maximum number of documents should be positive")
                return

        except ValueError:
            self.textEditConsole.append("> The maximum number of documents is not valid")
            return

        try:
            k_fold = int(self.lineEditDatasetKfold.text())

            if k_fold < 1:
                self.textEditConsole.append("> k should be positive")
                return

        except ValueError:
            self.textEditConsole.append("> k number is not valid")
            return

        dataset_path = self.lineEditDataset.text()

        # get data
        test_accuracy = TestAccuracy(dataset_path, max_nr_docs)
        results, names = test_accuracy.test_cross_val_score(k_fold, self.progressBar, self.textEditConsole)

        # clear any plot if it exists
        self.sentiment_display.clear_plot()

        # display data
        self.accuracy_display.plot_accuracy(results, names)
        self.tabWidgetData.setCurrentIndex(5)

    def __populate_treeview(self, data):
        """Populate the comments tab"""

        # get data
        videos_data = data.get_data_from_DB()

        self.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['element', 'value'])
        self.treeView.setModel(model)

        # parse data
        for video in videos_data:

            parent_elem = QStandardItem('video')
            parent_value = QStandardItem(video.get('title'))

            id_elem = QStandardItem('_id')
            id_value = QStandardItem(video.get('_id'))
            parent_elem.appendRow([id_elem, id_value])

            description_elem = QStandardItem('description')
            description_value = QStandardItem(video.get('description'))
            parent_elem.appendRow([description_elem, description_value])

            nr_likes_elem = QStandardItem('nr_likes')
            nr_likes_value = QStandardItem(video.get('nr_likes'))
            parent_elem.appendRow([nr_likes_elem, nr_likes_value])

            nr_dislikes_elem = QStandardItem('nr_dislikes')
            nr_dislikes_value = QStandardItem(video.get('nr_dislikes'))
            parent_elem.appendRow([nr_dislikes_elem, nr_dislikes_value])

            comments_elem = QStandardItem('comments')
            parent_elem.appendRow(comments_elem)

            comments = video.get("comments")

            for comment in comments:
                text_elem = QStandardItem('text')
                text_value = QStandardItem(comment.get('text'))
                comments_elem.appendRow([text_elem, text_value])

                cid_elem = QStandardItem('cid')
                cid_value = QStandardItem(comment.get('cid'))
                text_elem.appendRow([cid_elem, cid_value])

                time_elem = QStandardItem('time')
                time_value = QStandardItem(comment.get('time'))
                text_elem.appendRow([time_elem, time_value])

                author_elem = QStandardItem('author')
                author_value = QStandardItem(comment.get('author'))
                text_elem.appendRow([author_elem, author_value])

                nr_likes_elem = QStandardItem('nr_likes')
                nr_likes_value = QStandardItem(comment.get('nr_likes'))
                text_elem.appendRow([nr_likes_elem, nr_likes_value])

            model.appendRow([parent_elem, parent_value])


class SentimentAnalysisApplication(QMainWindow, UIMainWindow):
    """Main application -> initialises User Interface"""

    def __init__(self):
        """Class constructor"""

        QMainWindow.__init__(self, flags=QtCore.Qt.Window)
        UIMainWindow.__init__(self)
        self.setupUi(self)

