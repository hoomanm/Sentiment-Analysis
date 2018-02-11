# Sentiment Analysis on User Reviews

The goal of this project is to examine various supervised learning techniques in order to perform sentiment
classification. According to <a href="https://pdfs.semanticscholar.org/75d9/d183fd2c2cf6e0f7a957b9cf4fca38ed3253.pdf target="_blank"> Singh et al, </a>, the important supervised classification methods which are
used in the area of natural language processing and sentiment analysis are Naive Bayes Classifiers,
Support Vector Machines (SVMs), and Multilayer Perceptrons. We also used the Logistic Regression
classifier since it seems to have a good performance in text analysis applications according to different
articles, such as <a href="http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/" target="_blank"> Classifying text with bag of words </a>, and it also exhibited a high accuracy on both of our datasets. Below table shows the best F1 score of these classification techniques on each of our datasets after examining various combinations of the input parameters for feature extraction:

              Logistic Regression      Naive Bayes      SVM (Linear Kernel)      Multilayer Perceptron      Voting Classifier
              
IMDB Dataset:         90.03 %              86.88 %              89.34 %                   88.78 %                   90.10 %

UCI Dataset:
(Amazon, IMDB, Yelp)    82.46 %          83.13 %               83.73 %                  81.43 %                    83.96 %
