#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    Web crawler module
    @alexandru_grigoras
"""

# Libraries
import json
import multiprocessing
import queue
import time
import urllib.request
from urllib import robotparser

import lxml.html
import requests
from bs4 import BeautifulSoup
from lxml.cssselect import CSSSelector

from youtube_sentiment_analysis.modules.store import StoreData

# Constants
__all__ = ['WebCrawler']
__version__ = '1.0'
__author__ = 'Alexandru Grigoras'
__email__ = 'alex_grigoras_10@yahoo.com'
__status__ = 'release'

YOUTUBE_URL_SEARCH = "https://www.youtube.com/results?search_query="
YOUTUBE_PAGE_ARG = "&page="
YOUTUBE_URL = "https://www.youtube.com"
YOUTUBE_COMMENTS_URL = 'https://www.youtube.com/all_comments?v={youtube_id}'
YOUTUBE_COMMENTS_AJAX_URL = 'https://www.youtube.com/comment_ajax'
USER_AGENT = 'AG_SENTIMENT_ANALYSIS_BOT'


class Video:
    """Video class that contains the parameters of a video"""

    def __init__(self, title, link, description, likes=None, dislikes=None):
        """Class constructor"""
        self.__title = title
        self.__link = link
        self.__video_id = None
        self.__description = description
        self.__likes = likes
        self.__dislikes = dislikes

    def set_title(self, title):
        """Sets the title"""
        self.__title = title

    def set_link(self, link):
        """Sets the link"""
        self.__link = link

    def set_id(self, video_id):
        """Sets the id"""
        self.__video_id = video_id

    def set_description(self, description):
        """Sets the description"""
        self.__description = description

    def set_likes(self, likes):
        """Sets the likes"""
        self.__likes = likes

    def set_dislikes(self, dislikes):
        """Sets the dislikes"""
        self.__dislikes = dislikes

    def get_title(self):
        """Returns the title"""
        return self.__title

    def get_link(self):
        """Returns the link"""
        return self.__link

    def get_id(self):
        """Returns the id"""
        return self.__video_id

    def get_description(self):
        """Returns the description"""
        return self.__description

    def get_likes(self):
        """Returns the likes"""
        return self.__likes

    def get_dislikes(self):
        """Returns the dislikes"""
        return self.__dislikes

    def display(self, fp=None):
        """Displays the video data"""
        space = "         "
        try:
            print("\t> Title: " + self.__title, file=fp)
        except IOError:
            print(space + "> Invalid title!", file=fp)
        try:
            print(space + "Link: " + self.__link, file=fp)
        except IOError:
            print(space + "> Invalid link!", file=fp)
        try:
            print(space + "Description: " + self.__description, file=fp)
        except IOError:
            print(space + "> Invalid description!", file=fp)
        try:
            print(space + "Like: %s, Dislike: %s" % (self.__likes, self.__dislikes), file=fp)
        except IOError:
            print(space + "> No likes / dislikes!", file=fp)


class RobotParser:
    """Robot parser class to check the crawling rules on the domain and links"""

    def __init__(self):
        """Class constructor"""
        self.__rp = robotparser.RobotFileParser()

    def parse_domain(self):
        """Parse the domain for robot rules"""
        self.__rp.set_url(YOUTUBE_URL + "/robots.txt")
        self.__rp.read()

        r_rate = self.__rp.request_rate("*")
        if r_rate is not None:
            requests_nr = r_rate.requests
            request_sec = r_rate.seconds
            #print("> Parameters: ")
            #print("\t - request-rate: " + str(requests_nr) + "/" + str(request_sec) + "s")

        # TO DO: add other parameters to test

    def can_extract(self, link):
        """Checks the link to validate the crawling permission"""
        return self.__rp.can_fetch("*", link)


class Downloader(multiprocessing.Process):
    """Worker for downloading data from every video in the list"""

    def __init__(self, name, keyword, videos_list):
        multiprocessing.Process.__init__(self)
        self.__keyword = keyword
        self.__videos_list = videos_list
        self.__exception = None
        self.__mongo_conn = None

        print("> Initialize worker " + name + " with " + str(len(videos_list)) + " videos")

    def run(self):
        self.__mongo_conn = StoreData(self.__keyword, store=True)

        try:
            # search every video for metadata
            for video in self.__videos_list:
                try:
                    print("> Crawling " + video.get_title())
                    self.__video_process(video)
                except AttributeError:
                    print("> Extracted data from video is invalid (AttributeError)!")
                except IndexError:
                    print("> Extracted data from video is invalid (IndexError)!")
        except Exception as e:
            self.__exception = e

    def __video_process(self, video):
        """Process every video to find links"""
        video_id_path = video.get_link()
        video_id = video_id_path.replace("/watch?v=", "")
        url_video = YOUTUBE_URL + video_id_path

        rp = RobotParser()

        rp.parse_domain()

        if rp.can_extract(url_video) is False:
            print("> Page cannot be crawled: " + url_video)
            return

        headers = {'User-Agent': USER_AGENT}
        req = urllib.request.Request(url_video, headers=headers)
        search_content = urllib.request.urlopen(req)
        search_content_html = BeautifulSoup(search_content, 'lxml')

        try:
            like = search_content_html.findAll('button', {"class": "like-button-renderer-like-button"})
            likes = like[0].span.text
        except IndexError:
            likes = 0
        try:
            dislike = search_content_html.findAll('button', {"class": "like-button-renderer-dislike-button"})
            dislikes = dislike[0].span.text
        except IndexError:
            dislikes = 0

        # create a video
        video.set_link(url_video)
        video.set_likes(likes)
        video.set_dislikes(dislikes)
        video.set_id(video_id)

        if video_id_path.find("channel") is -1 and video_id_path.find("user") is -1:
            self.__metadata_extractor(video)
        elif video_id_path.find("channel") is not -1:
            print("> " + video.get_title() + " is a channel")
        elif video_id_path.find("user") is not -1:
            print("> " + video.get_title() + " is a user")
        else:
            print("> " + video.get_title() + " is unknown")

    def __metadata_extractor(self, video):
        """Extracts the data from video"""
        count = self.__download_comments(video)

        print('> Downloading ' + str(count) + ' comments for video: ', video.get_title(), ' (', video.get_id(), ')')

    def __download_comments(self, video=None, sleep=0):
        """Extract comments from video"""
        session = requests.Session()
        session.headers['User-Agent'] = USER_AGENT

        # get Youtube page with initial comments
        response = session.get(YOUTUBE_COMMENTS_URL.format(youtube_id=video.get_id()))
        html = response.text
        reply_comments = self.__extract_reply_comments(html)

        nr_comments = 0

        nr_comments += self.__extract_comments(html, video)

        page_token = self.__find_token(html, 'data-token')
        session_token = self.__find_token(html, 'XSRF_TOKEN', 4)

        first_iteration = True

        # get remaining comments
        while page_token:
            data = {'video_id': video.get_id(),
                    'session_token': session_token}

            params = {'action_load_comments': 1,
                      'order_by_time': True,
                      'filter': video.get_id()}

            if first_iteration:
                params['order_menu'] = True
            else:
                data['page_token'] = page_token

            response = self.__ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL, params, data)
            if not response:
                break

            page_token, html = response

            reply_comments += self.__extract_reply_comments(html)
            nr_comments += self.__extract_comments(html, video)

            first_iteration = False
            time.sleep(sleep)

        # get replies
        for cid in reply_comments:
            data = {'comment_id': cid,
                    'video_id': video.get_id(),
                    'can_reply': 1,
                    'session_token': session_token}

            params = {'action_load_replies': 1,
                      'order_by_time': True,
                      'filter': video.get_id(),
                      'tab': 'inbox'}

            response = self.__ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL, params, data)
            if not response:
                break

            _, html = response

            nr_comments += self.__extract_comments(html, video)

            time.sleep(sleep)

        return nr_comments

    @staticmethod
    def __extract_reply_comments(html):
        """Get comments from replies"""
        tree = lxml.html.fromstring(html)
        sel = CSSSelector('.comment-replies-header > .load-comments')
        return [i.get('data-cid') for i in sel(tree)]

    def __extract_comments(self, html, video):
        """Extracts comments from html using CSSSelector to find specific classes"""
        tree = lxml.html.fromstring(html)
        item_sel = CSSSelector('.comment-item')
        text_sel = CSSSelector('.comment-text-content')
        time_sel = CSSSelector('.time')
        author_sel = CSSSelector('.user-name')
        like_sel = CSSSelector('.like-count')

        nr_comments = 0

        for item in item_sel(tree):
            self.__mongo_conn.write(video,
                                    item.get('data-cid'),
                                    text_sel(item)[0].text_content(),
                                    time_sel(item)[0].text_content().strip(),
                                    author_sel(item)[0].text_content(),
                                    like_sel(item)[0].text_content(),
                                    )
            nr_comments += 1

        return nr_comments

    @staticmethod
    def __find_token(html, key, num_chars=2):
        """Find start and end position of a key"""
        begin = html.find(key) + len(key) + num_chars
        end = html.find('"', begin)

        return html[begin: end]

    @staticmethod
    def __ajax_request(session, url, params, data, retries=1, sleep=0):
        """Ajax request to scroll page"""
        for _ in range(retries):
            response = session.post(url, params=params, data=data)
            if response.status_code == 200:
                response_dict = json.loads(response.text)
                return response_dict.get('page_token', None), response_dict['html_content']
            else:
                time.sleep(sleep)

    def get_exception(self):
        """Returns the generated exception"""
        return self.__exception


class WebCrawler:
    """Search on youtube by chosen keyword and find all videos to download comments"""

    def __init__(self, keyword, nr_videos, crawl_delay):
        """Class constructor"""

        self.__keyword = keyword
        self.__nr_videos = nr_videos
        self.__crawl_delay = crawl_delay
        self.__videos_queue = queue.Queue(maxsize=1000)

    def run(self, console):
        """Method that runs the web crawler on YouTube with the specified keyword for search"""

        # get the starting time
        start_time = time.time()

        # start the main processing based on arguments
        self.__search_pages(console)

        # get the finish time and calculate the script execution time
        end_time = time.time()
        elapsed_time = end_time - start_time
        console.append("> Data extracted in " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + " seconds")

        return True

    def __search_pages(self, console):
        """Searches into YouTube results pages to find videos"""

        # beginning search__get_videos
        console.append("> Searching for: " + self.__keyword)

        # check limit of pages and videos
        if self.__nr_videos < 1:
            console.append("> The number of videos should be > 0")
        elif self.__nr_videos == 1:
            console.append("> Limit the search to: " + str(self.__nr_videos) + " video")
        else:
            console.append("> Limit the search to: " + str(self.__nr_videos) + " videos")

        extracted_videos = 0
        max_nr_pages = 50
        current_page = 0

        # add the videos to the queue
        while extracted_videos < self.__nr_videos and current_page <= max_nr_pages:
            current_page += + 1
            url = YOUTUBE_URL_SEARCH + self.__keyword.replace(" ", "%20") + YOUTUBE_PAGE_ARG + str(current_page)

            try:
                value = self.__get_videos(url, self.__nr_videos, console)
                if value is -1:
                    console.append("> There isn't an internet connection! Please connect to the internet to get data from videos!")
                    print("no internet")
                extracted_videos += value
            except Exception:
                console.append("> There isn't an internet connection! Please connect to the internet to get data from videos!")
                return

            time.sleep(self.__crawl_delay)

        # get the number of threads
        nr_threads = multiprocessing.cpu_count()

        # page threads list
        page_processes = []

        # calculate the number of videos for each thread to be processed
        nr_videos_in_queue = self.__videos_queue.qsize()
        console.append("> Number of videos found: " + str(nr_videos_in_queue))

        if nr_videos_in_queue == 0:
            print("> Cannot run crawling with no videos!")
            console.append("> Cannot run crawling with no videos!")

        # create the workers to process the videos
        if nr_videos_in_queue <= nr_threads:
            for i in range(0, nr_videos_in_queue, 1):
                videos_list = []
                if self.__videos_queue.empty() is False:
                    videos_list.append(self.__videos_queue.get())
                process = Downloader(str(i), self.__keyword, videos_list)
                page_processes.append(process)
        else:
            video_per_thread = int(nr_videos_in_queue / nr_threads)
            remaining_videos = nr_videos_in_queue % nr_threads

            for i in range(0, nr_threads, 1):
                videos_list = []
                index = 0

                if remaining_videos > 0:
                    total_videos = video_per_thread + 1
                    remaining_videos -= 1
                else:
                    total_videos = video_per_thread

                while self.__videos_queue.empty() is False and index < total_videos:
                    videos_list.append(self.__videos_queue.get())
                    index += 1
                process = Downloader(str(i), self.__keyword, videos_list)
                page_processes.append(process)

        # start the workers
        for process in page_processes:
            process.start()

        # wait for each worker to finish the processing
        for process in page_processes:
            process.join()

        # check if there where any errors on workers
        for t in page_processes:
            e = t.get_exception()
            if e:
                console.append("> Error on process:" + e)

    def __get_videos(self, url, max_nr_videos=None, console=None):
        """Finds the videos in the selected YouTube page"""

        # set header for request
        headers = {'User-Agent': USER_AGENT}
        req = urllib.request.Request(url, headers=headers)

        try:
            # send request
            try:
                search_result = urllib.request.urlopen(req)
            except urllib.error.URLError:
                print('Cannot make request')
                return -1
            
            soup = BeautifulSoup(search_result, 'lxml')
            description = soup.findAll('div', {"class": "yt-lockup-description"})
            title_link = soup.findAll('a', {"class": "yt-uix-tile-link"})

            # check the number of videos
            if len(title_link) == 0:
                return 0

            if max_nr_videos:
                selected_nr_videos = max_nr_videos
            else:
                selected_nr_videos = len(title_link)

            # search every video for metadata
            for video in range(0, selected_nr_videos, 1):
                try:
                    # put the video in the queue
                    current_video = Video(title_link[video]['title'],
                                          title_link[video]['href'],
                                          description[video].text)
                    self.__videos_queue.put(current_video)
                except AttributeError:
                    console.append("> Video cannot be put to queue (AttributeError)!")
                except IndexError:
                    console.append("> Video cannot be put to queue (IndexError)!")

            # returns the number of the videos found
            return len(title_link)

        except urllib.error.HTTPError:
            console.append("> HTTP request error: Too many requests")
            return 0
