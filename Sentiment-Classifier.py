import sys
import os
import scipy.sparse
import sklearn.naive_bayes as NB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import text
from porter2stemmer import Porter2Stemmer


########### Reading the data and the labels ########### 
Imdb_train_data = []
Imdb_train_labels = []
Imdb_test_data = []
Imdb_test_labels = []
UCI_train_data = []
UCI_train_labels = []

########### IMDB Reviews ########### 
dirname = "./aclImdb/train/neg"
for fname in os.listdir(dirname):
    with open(os.path.join(dirname, fname), 'r') as f:
        content = f.read()
        Imdb_train_data.append(content)
        Imdb_train_labels.append("neg")

dirname = "./aclImdb/train/pos"
for fname in os.listdir(dirname):
    with open(os.path.join(dirname, fname), 'r') as f:
        content = f.read()
        Imdb_train_data.append(content)
        Imdb_train_labels.append("pos")

dirname = "./aclImdb/test/neg"
for fname in os.listdir(dirname):
    with open(os.path.join(dirname, fname), 'r') as f:
        content = f.read()
        Imdb_test_data.append(content)
        Imdb_test_labels.append("neg")

dirname = "./aclImdb/test/pos"
for fname in os.listdir(dirname):
    with open(os.path.join(dirname, fname), 'r') as f:
        content = f.read()
        Imdb_test_data.append(content)
        Imdb_test_labels.append("pos")


########### UCI: Amazon, IMDB, Yelp Reviews ###########
with open("UCI-sentiment-labelled-sentences/amazon_cells_labelled.txt", 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content] 

for review in content:
    UCI_train_data.append(review.split("\t")[0])
    UCI_train_labels.append(review.split("\t")[1])

with open("UCI-sentiment-labelled-sentences/imdb_labelled.txt", 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content] 

for review in content:
    UCI_train_data.append(review.split("\t")[0])
    UCI_train_labels.append(review.split("\t")[1])

with open("UCI-sentiment-labelled-sentences/yelp_labelled.txt", 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content] 

for review in content:
    UCI_train_data.append(review.split("\t")[0])
    UCI_train_labels.append(review.split("\t")[1])



new_words = []
new_review = ""
stemmer = Porter2Stemmer()

############ Creating feature vectors ###########
vectorizer1 = text.TfidfVectorizer(min_df=2,
                             #max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True,
                             max_features = 30000,
                             ngram_range = (1, 2))
                             #stop_words = text.ENGLISH_STOP_WORDS)

vectorizer2 = text.TfidfVectorizer(min_df=2,
                             #max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True,
                             # max_features = 10000,
                             ngram_range = (1, 2))
                             #stop_words = text.ENGLISH_STOP_WORDS)

Imdb_train_vectors = vectorizer1.fit_transform(Imdb_train_data)
UCI_train_vectors = vectorizer2.fit_transform(UCI_train_data)

Imdb_test_vectors = vectorizer1.transform(Imdb_test_data)

print "IMDB Features Size: ", len(Imdb_train_vectors.toarray()[0])
print "UCI Features Size: ", len(UCI_train_vectors.toarray()[0])
print "\n"


######******************* Calssification Models *******************######

###################### Logistic Regression ####################
logistic_reg = LR()

scores = cross_val_score(logistic_reg, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')
#scores = cross_val_score(log_reg, train_vectors, train_labels, cv=5)

logistic_reg.fit(Imdb_train_vectors, Imdb_train_labels)
logistic_reg_pred = logistic_reg.predict(Imdb_test_vectors)

print "Logistic Regression F1 Score on IMDB dataset: ", f1_score(Imdb_test_labels, logistic_reg_pred, average='macro')
print "Logistic Regression F1 Score on UCI dataset: ", scores.mean()
print "\n"
# print "Logistic Regression Mean Accuracy on IMDB dataset: ", logistic_reg.score(Imdb_test_vectors, Imdb_test_labels)
# print classification_report(Imdb_test_labels, logistic_reg_pred)


###################### Multilayer Perceptron ######################
ML_perceptron = MLPClassifier()
UCI_scores = cross_val_score(ML_perceptron, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')
ML_perceptron.fit(Imdb_train_vectors, Imdb_train_labels)
ML_prediction = ML_perceptron.predict(Imdb_test_vectors)

print "ML Perceptron F1 Score on IMDB dataset: ", f1_score(Imdb_test_labels, ML_prediction, average='macro')
print "ML Perceptron F1 Score on UCI dataset: ", UCI_scores.mean()
print "\n"
# print classification_report(test_labels, ML_prediction)

###################### SVM, kernel=linear ######################
classifier_liblinear = svm.LinearSVC()
UCI_scores = cross_val_score(classifier_liblinear, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')

classifier_liblinear.fit(Imdb_train_vectors, Imdb_train_labels)
classifier_liblinear_pred = classifier_liblinear.predict(Imdb_test_vectors)

print "SVM F1 Score on IMDB dataset: ", f1_score(Imdb_test_labels, classifier_liblinear_pred, average='macro')
print "SVM F1 Score on UCI dataset: ", UCI_scores.mean()
print "\n"
#print "Classification Report for SVM on IMDB dataset:\n", classification_report(Imdb_test_labels, classifier_liblinear_pred)


###################### Naive Bayes ######################
bnb = NB.MultinomialNB()
UCI_scores = cross_val_score(bnb, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')
bn_pred = bnb.fit(Imdb_train_vectors.toarray(), Imdb_train_labels).predict(Imdb_test_vectors.toarray())

print "Naive Bayes F1 Score on IMDB dataset: ", f1_score(Imdb_test_labels, bn_pred, average='macro')
print "Naive Bayes F1 Score on UCI dataset: ", UCI_scores.mean()
print "\n"
#print "Classification Report for Naive Bayes on IMDB dataset:\n", classification_report(Imdb_test_labels, y_pred)


###################### Voting Classifier ######################
voting_clf = VotingClassifier(estimators=[('nb', bnb), ('lg1', logistic_reg), ('svc', classifier_liblinear), ('mlp', ML_perceptron)],
                       voting='hard', weights=[1,1,1,1])
# voting_clf = VotingClassifier(estimators=[('nb', bnb), ('lg1', logistic_reg), ('svc', classifier_liblinear)],
#                       voting='hard', weights=[1,1,1])
UCI_scores = cross_val_score(voting_clf, UCI_train_vectors, UCI_train_labels, cv=10, scoring='f1_macro')
voting_clf.fit(Imdb_train_vectors, Imdb_train_labels)
voting_clf_pred = voting_clf.predict(Imdb_test_vectors)

print "Voting Classifier F1 Score on IMDB dataset: ", f1_score(Imdb_test_labels, voting_clf_pred, average='macro')
print "Voting Classifier F1 Score on UCI dataset: ", UCI_scores.mean()
#print "Classification Report for Voting Classifier on IMDB dataset:\n", classification_report(Imdb_test_labels, voting_clf)






