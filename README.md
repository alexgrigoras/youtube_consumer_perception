# Consumer perception of online multimedia

## Description
This is a bachelor's degree application. It was created to analyze Youtube videos' metadata to create statistics about a specific key phrase search.
It can determine the impact of user comments on videos. It is helpful for the content creators of the Youtube platform. 

<img src="./docs/images/users_perception.png" alt="Consumer perception image" width="500">

## Components
The main components of the application are:
1. Web Crawler
	- search videos on youtube based on a keyphrase
1. Data Storage
	- store data to mongoDB
1. Data Processing
	- preprocess data to remove noise
1. Data Analysis
	- analyse extracted data
	- machine learning classifiers:
		* Multinomial Naive Bayes
		* Logistic Regression
		* Support Vector Machines
	- lexicon based dictionaries:
		* VADER
		* ANEW
	- voting system (using all classifiers)
1. User interface
	- interface for user to search and view data
1. Data display
	- create the graphs for displaying data

## Structure
```
youtube_sentiment_analysis
├── docs
|   └── images
|       └── users_perception.png
├── venv
├── youtube_sentiment_analysis
|   ├── data
|   |   └── classifiers
|   |   |   ├── logistic_regression.pickle
|   |   |   ├── multinomial_naive_bayes.pickle
|   |   |   └── nu_svc.pickle
|   ├── modules
|   |   ├── __init__.py
|   |   ├── accuracy.py
|   |   ├── analysis.py
|   |   ├── crawler.py
|   |   ├── display.py
|   |   ├── interface.py
|   |   ├── process.py
|   |   ├── store.py
|   |   ├── training.py
|   |   └── vote_classifier.py
|   ├── __init__.py
|   └── __main__.py
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Accuracy of the classifiers

### Dataset with 10,000 files from IMDB
| Classifier                | Accuracy (%)  | Standard deviation |
| ------------------------- | ------------- | ------------------ |
| Multinomial NB            | 85.66         | 0.0065		 |
| Logistic Regression       | 87.30         | 0.0107             |
| Nu SVC                    | 88.23         | 0.0078             |
| VADER			    | 69.36	    | 0.1638		 |
| ANEW			    | 67.54	    | 0.1313		 |
| VOTING		    | 88.88	    | 0.0977		 |

*Training time was obtained on a virtual machine with Debian 9.6 OS, with the specifications: 
(CPU Intel Core i7-2600 (4 cores), RAM 8GB DDR3, GPU AMD Radeon RX580)

## Chart Legend
 - horizontal axis 	-> valence (from voting system)
 - vertical axis	-> arousal (from ANEW)
 - Colour       	-> the confidence of classifiers
 - Size         	-> the number of likes

## New features to implement
 - bigger dataset (more than 50000 files containing comments);
 - scatter point with details on hover;
 - on y-axis will be added activity of the user that write the comment;
 - points size will be based on their likes.
 
## Resources
### 1. Academic papers
- [1] Flora Amato, Aniello Castiglione, Fabio Mercorio, Mario Mezzanzanica, Vincenzo Moscato, Antonio Picariello, Giancarlo Sperlì, „Multimedia story creation on social networks”, Elsevier, vol. 86, pp. 412-420, 2018. 
- [2] Bo Han, „Improving the Utility of Social Media with Natural Language Processing”, The University of Melbourne, PhD Thesis, pp. 1-198, 2014. 
- [3] Androniki Sapountzi, Kostas E. Psannis, „Social networking data analysis tools & challenges”, Elsevier, vol. 86, pp. 893-913, 2016. 
- [4] Ratab Gulla, Umar Shoaiba, Saba Rasheed, Washma Abid, Beenish Zahoor, „Pre Processing of Twitter's Data for Opinion Mining in Political Context”, Elsevier, vol. 96, pp. 1560-1570, 2016. 
- [5] Christopher G. Healey, Tweet Sentiment Visualization [Online], Disponibil la adresa: https://www.csc2.ncsu.edu/faculty/healey/tweet_viz/, Accesat: 2018. 
- [6] Google, Natural Language [Online], Disponibil la adresa: https://cloud.google.com/naturallanguage/, Accesat: 2019. 
- [7] Microsoft, Text Analytics [Online], Disponibil la adresa: https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/, Accesat: 2019. 
- [8] R. Piryani, D. Madhavi, V.K. Singh, „Analytical mapping of opinion mining and sentiment analysis research during 2000–2015”, Elsevier, vol. 53, pp. 122-150, 2016. 
- [9] Stuart J. Russell, Peter Norvig, „Artificial Intelligence A Modern Approach Third Edition”, Pearson Education, Inc., 2010. 
- [10] Diksha Khurana, Aditya Koli, Kiran Khatter and Sukhdev Singh, „Natural Language Processing: State of The Art, Current Trends and Challenges”, ResearchGate, Vol. 1, pp. 1-25, 2017. 
- [11] Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze, „An Introduction to Information Retrieval”, Cambridge University Press, 2009. 
- [12] Bing Liu, „Sentiment Analysis and Opinion Mining”, Morgan&Claypool, 2012. 
- [13] Bing Liu, „Sentiment Analysis and Subjectivity”, ResearchGate, vol. 26, pp. 627-666, 2010. 
- [14] Judith Hurwitz, Daniel Kirsch, „Machine Learning For Dummies”, John Wiley & Sons, Inc, 2018. 
- [15] Peter Jeffcock (ORACLE), What's the Difference Between AI, Machine Learning, and Deep Learning? [Online], Disponibil la adresa: https://blogs.oracle.com/bigdata/difference-ai-machinelearning-deep-learning, Accesat: 2019. 
- [16] K. Ming Leung, „Naive Bayesian Classifier”, POLYTECHNIC UNIVERSITY, University Course, pp. 1-16, 2007. 
- [17] Sona Taheri, Musa Mammadov, „Learning the Naive Bayes Classifier with optimization models”, International Journal of Applied Mathematics and Computer Science, vol. 23, pp. 787–795, 2013. 
- [18] Rennie, Jason D. M. and Shih, Lawrence and Teevan, Jaime and Karger, David R., „Tackling the Poor Assumptions of Naive Bayes Text Classiers”, AAAI Press, ICML'03, pp. 616-623, 2003. 
- [19] Harry Zhang, „The Optimality of Naive Bayes”, Proceedings of the , Seventeenth International Florida Artificial Intelligence Research Society Conference, pp. 1-6, 2004. 
- [20] Scikit Learn, Generalized Linear Models [Online], Disponibil la adresa: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression, Accesat: 2019. 
- [21] NCSS, „Logistic Regression”, NCSS Statistical Software, Capitolul 321, pp. 1-69, 2019. 
- [22] Longjian Liu, „Biostatistical Basis of Inference in Heart Failure Study”, Longjian Liu, Heart Failure: Epidemiology and Research Methods, pp. 43-82, 2018. 
- [23] Andrew Ng, „Support Vector Machines”, Stanford, CS229: Machine Learning, pp. 1-25, 2018.
- [24] Scikit Learn, Support Vector Machines [Online], Disponibil la adresa: https://scikit-learn.org/stable/modules/svm.html#svm-kernels, Accesat: 2019. 
- [25] Maite Taboada, „Sentiment Analysis: An Overview from Linguistics”, Simon Fraser University, Annual Review of Applied Linguistics, pp. 1-52, 2016. 
- [26] Prabu Palanisamy, Vineet Yadav, Harsha Elchuri, „Serendio: Simple and Practical lexicon based approach to Sentiment Analysis”, Association for Computational Linguistics, Second Joint Conference on Lexical and Computational Semantics, pp. 543–548, 2013. 
- [27] C.J. Hutto, Eric Gilbert, „VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text”, Proceedings of the , Eighth International AAAI Conference on Weblogs and Social Media, pp. 1-10, 2015. 
- [28] James A Russell, „A Circumplex Model of Affect”, Journal of Personality and Social Psychology, Vol. 39, pp. 1161-1178, 1980. 
- [29] Margaret M. Bradley, Peter J. Lang, „Affective Norms for English Words (ANEW): Instruction Manual and Affective Ratings”, NIMH Center for Emotion and Attention, Vol. 1, pp. 1-48, 1999. 
- [30] Amy Beth Warriner, Victor Kuperman, Marc Brysbaert, „Norms of valence, arousal, and dominance for 13,915 English lemmas”, Behavior Research Methods, Vol. 45, pp. 1191–1207, 2013. 
- [31] IBM, Three-tier architectures [Online], Disponibil la adresa: https://www.ibm.com/support/knowledgecenter/en/SSAW57_9.0.0/com.ibm.websphere.nd.multipla tform.doc/ae/covr_3-tier.html, Accesat: 2019. 
- [32] Carlos Castillo, „Effective Web Crawling”, University of Chile, PhD, pp. 1-164, 2004. 
- [33] Gerald Petz, Michał Karpowicz, Harald Fürschuß, Andreas Auinger, Václav Strítesky, Andreas Holzinger, „Reprint of: Computational approaches for mining user’sopinions on the Web 2.0”, Elsevier, vol. 51, pp. 510-519, 2015. 
- [34] Sam Henry, Clint Cuffy, Bridget T. McInnes, „Vector representations of multi-word terms for semantic relatedness”, Journal of Biomedical Informatics, Vol. 77, pp. 111-119, 2017. 
- [35] Gareth James, „Majority Vote Classifiers: Theory and applications”, STANFORD UNIVERSITY, PhD Thesis, pp. 1-98, 1998. 
- [36] Dave Kuhlman, „A Python Book: Beginning Python, Advanced Python and Python Exercises”, Platypus Global Media, 2013. 
- [37] Shiliang Sun, Chen Luo, Junyu Chen, „A review of natural language processing techniques for opinion mining systems”, Elsevier, vol. 36, pp. 10-25, 2016. 
- [38] Steven Bird, Ewan Klein, Edward Loper, „Natural Language Processing with Python”, O’Reilly Media, Inc., 2009. 
- [39] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Perrot, Édouard Duchesnay, „Scikit-learn: Machine Learning in Python”, Journal of Machine Learning Research, 12, pp. 2825-2830, 2011. 
- [40] PyQt, PyQt Reference Guide [Online], Disponibil la adresa: https://www.riverbankcomputing.com/ static/Docs/PyQt5/, Accesat: 2019. 
- [41] Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T.  and  Huang, Dan and Ng, Andrew Y. and Potts, Christopher, „Learning Word Vectors for Sentiment Analysis”, Association for Computational Linguistics, vol. 1, pp. 142-150, 2011. 
- [42] Payam Refaeilzadeh, Lei Tang, Huan Liu, „Cross-Validation”, Springer, Encyclopedia of Database Systems, pp. 532-538, 2009. 
- [43] Aliaksei Severyn, Alessandro Moschitti, Olga Uryupina, Barbara Plank, Katja Filippova, „Multilingual opinion mining on YouTube”, Elsevier, vol. 52, pp. 46-60, 2015.

### 2. GitHub
 - [youtube-comment-downloader](https://github.com/egbertbouman/youtube-comment-downloader)
 - [VADER-Sentiment-Analysis](https://github.com/cjhutto/vaderSentiment)
 
### 3. Python Libraries
 - [VADER](https://www.nltk.org/_modules/nltk/sentiment/vader.html): Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
 - [NLTK](https://www.nltk.org/)
 - [SciKit-learn](https://scikit-learn.org/stable)
 - [WordCloud](https://amueller.github.io/word_cloud/)
 
### 4. Dataset
 - [Kaggle](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset): Potts, Christopher. 2011. On the negativity of negation. In Nan Li and David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20, 636-659.
 
### 5. Tutorials
 - [Pythonprogramming - Natural language processing](https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/)
 - [DZone - Simple Sentiment Analysis With NLP](https://dzone.com/articles/simple-sentiment-analysis-with-nlp)
 
## Existing products
### 1. Academic
 - [Tweet Sentiment Visualization](https://www.csc2.ncsu.edu/faculty/healey/tweet_viz/tweet_app/)

### 2. Commercial
 - [Google - Cloud Natural Language](https://cloud.google.com/natural-language/)
 - [Microsoft - Text Analytics API](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/overview) 

## License
The application is licensed under MIT License. It is made for academic use and is the subject of a bachelor's degree.
