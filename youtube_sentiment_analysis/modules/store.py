#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Store data module
    @alexandru_grigoras
"""

# Libraries
import pymongo

# Constants
__all__ = ['StoreData']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'

DATABASE_NAME = "sentiment_analysis"


class StoreData:
    """Class for storing the data to temporary files or MongoDB database"""

    def __init__(self, keyword=None, store=True, console=None):
        """Class constructor"""

        try:
            self.__my_client = pymongo.MongoClient("mongodb://localhost:27017/")
        except Exception:
            if console:
                console.append("> MongoDB database connection is closed or MongoDB is not installed!")
            return

        self.__db_list = self.__my_client.list_database_names()

        if DATABASE_NAME not in self.__db_list:
            if console:
                console.append("> Database " + DATABASE_NAME + " does not exists. Creating it!")

        self.__my_db = self.__my_client["sentiment_analysis"]

        self.__col_list = self.__my_db.list_collection_names()
        if keyword:
            if keyword not in self.__col_list:
                if not store:
                    if console:
                        console.append("> Collection does not exist! Extract data first!")
                    exit()
                if console:
                    console.append("> Collection " + keyword + " does not exists. Creating it!")

            self.__my_col = self.__my_db[keyword]

    def write(self, video, cid, text, time, author, nr_likes):
        """Write data on the database"""

        my_query = {
            '_id': video.get_id() if video else "",
            'title': video.get_title() if video else "",
            'description': video.get_description() if video else "",
            'nr_likes': video.get_likes() if video else "",
            'nr_dislikes': video.get_dislikes() if video else "",
        }
        new_values = {
            "$addToSet":
                {
                    'comments':
                        {
                            'cid': cid,
                            'text': text,
                            'time': time,
                            'author': author,
                            'nr_likes': nr_likes,
                        }
                }
        }
        self.__my_col.update_one(my_query, new_values, upsert=True)

    def read(self):
        """Read data from mongodb database"""

        return self.__my_col.find()

    def get_collections(self):
        """Get the collections from database"""

        return self.__col_list
