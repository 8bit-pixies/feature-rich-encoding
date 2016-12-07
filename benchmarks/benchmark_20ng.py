from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from FeatureRichEncoding import FeatureRichEncodingVectorizer

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

print "Training with only TFIDF"

vectorizer = TfidfVectorizer()

vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

clf = MultinomialNB(alpha=.01)
clf.fit(vectors_train, newsgroups_train.target)

pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')

print "Training with default Feature Rich Encoding"

richVectorizer = FeatureRichEncodingVectorizer

print "\ttransforming training set with rich feature set"
richvectors_train = richVectorizer.fit_transform(newsgroups_train.data)
print "\ttransforming test set with rich feature set"
richvectors_test = richVectorizer.transform(newsgroups_test.data)

richclf = MultinomialNB(alpha=.01)
richclf.fit(richvectors_train, newsgroups_train.target)

pred = richclf.predict(richvectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')
