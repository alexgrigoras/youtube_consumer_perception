from pip._internal.download import PipSession
from pip._internal.req import parse_requirements
from setuptools import setup, find_packages

from youtube_sentiment_analysis import __version__ as version

requirements = [
    str(req.req) for req in parse_requirements('requirements.txt', session=PipSession())
]

setup(
    name="youtube_sentiment_analysis",
    version=version,
    author="Alexandru Grigoras",
    author_email="alex_grigoras_10@yahoo.com",
    description="Youtube Sentiment Analysis",
    url="https://bitbucket.org/grigorasalex/youtube_sentiment_analysis/src/master/",
    packages=find_packages(),
    keywords='youtube search sentiment analysis',
    install_requires=requirements,
    zip_safe=True,
    classifiers=[
        'Development Status :: 1.0 - Release',
        "Programming Language :: Python :: 3.6",
        "Artificial Intelligence :: Natural Language Processing",
        "Crawler :: Youtube metadata crawler",
        "Operating System :: OS Independent",
    ],
)