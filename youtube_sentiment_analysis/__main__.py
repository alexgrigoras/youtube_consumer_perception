#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Youtube Sentiment Analysis - Main module
    @alexandru_grigoras
"""

# Libraries
import sys

from PyQt5.QtWidgets import QApplication

from youtube_sentiment_analysis.modules.interface import SentimentAnalysisApplication

# Constants
__all__ = []
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'

if __name__ == "__main__":
    """Main function that starts the application"""

    app = QApplication(sys.argv)
    window = SentimentAnalysisApplication()
    window.show()
    sys.exit(app.exec_())
