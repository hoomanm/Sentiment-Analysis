# Sentiment Analysis on User Reviews

The goal of this project is to examine various supervised learning techniques in order to perform sentiment
classification. We use two datasets of user reviews. The first one is <a href="http://ai.stanford.edu/~amaas/data/sentiment/"> Large Movie Review Dataset </a> which contains 50,000 reviews split evenly into 25k train and 25k test sets. The second one is <a href="https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences"> UCI Sentiment Labeled Sentences </a> which include 500 positive and 500 negative sentences from each of the Amazon, IMDB, and Yelp websites.
According to <a href="https://pdfs.semanticscholar.org/75d9/d183fd2c2cf6e0f7a957b9cf4fca38ed3253.pdf" target="_blank"> Singh et al.</a>, the important supervised classification methods which are used in the area of natural language processing and sentiment analysis are Naive Bayes Classifiers, Support Vector Machines (SVMs), and Multilayer Perceptrons. We also used the Logistic Regression classifier since it seems to have a good performance in text analysis applications according to different
articles, such as <a href="http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/" target="_blank"> Classifying text with bag of words</a>. This classifier also exhibited a high accuracy on both of the datasets. The table below shows the best F1 score of these classification techniques using each of our datasets after examining various combinations of the input parameters for feature extraction:

| Dataset | Logistic Regression | Naive Bayes | SVM (Linear Kernel) | Multilayer Perceptron | Voting Classifier |
|-------- | ------------------- | ----------- | ------------------- | --------------------- | ----------------- |    
| IMDB Dataset |   90.03 %      |   86.88 %   |        89.34 %      |      88.78 %          |      90.10 %      |
| UCI Dataset  |    82.46 %     |   83.13 %   |        83.73 %      |      81.43 %          |      83.96 %      |
