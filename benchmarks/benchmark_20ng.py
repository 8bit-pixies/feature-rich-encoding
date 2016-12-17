from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif, SelectPercentile, VarianceThreshold

from FeatureRichEncoding import FeatureRichEncoding
from sklearn.pipeline import FeatureUnion, Pipeline   

import pickle

categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print "Training with only TFIDF"
vectorizer = TfidfVectorizer()

vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vFS = Pipeline([('vs', VarianceThreshold()), 
                ('sp', SelectPercentile(mutual_info_classif, percentile=1))])
vectorsFS_train = vFS.fit_transform(vectors_train, newsgroups_train.target)

vectors_test = vectorizer.transform(newsgroups_test.data)
vectorsFS_test = vFS.transform(vectors_test)

clf = MultinomialNB()
clf.fit(vectorsFS_train, newsgroups_train.target)

pred = clf.predict(vectorsFS_test)
print "Base line performance is {}\n".format(metrics.f1_score(newsgroups_test.target, pred, average='macro'))
#print "Base line performance is {}".format(metrics.f1_score(newsgroups_train.target, clf.predict(vectors_train), average='macro'))

print "Training with default Feature Rich Encoding"

richVectorizer = FeatureUnion([('w2v', Pipeline([('w2v', FeatureRichEncoding(max_vocab_size=10**5, batch_words=10**3)), ('minmax', MinMaxScaler())])), 
                               ('pos', Pipeline([('pos', FeatureRichEncoding(mode='pos', max_vocab_size=10**5, batch_words=10**3)), ('minmax', MinMaxScaler())])),
                               ('ner', Pipeline([('ner', FeatureRichEncoding(mode='ner', max_vocab_size=10**5, batch_words=10**3)), ('minmax', MinMaxScaler())])),
                               ('tfidf', TfidfVectorizer())])

print "\ttransforming training set with rich feature set"
richvectors_train = richVectorizer.fit_transform(newsgroups_train.data)

frFS = Pipeline([('vs', VarianceThreshold()), 
                ('sp', SelectPercentile(mutual_info_classif, percentile=1))])
richvectorsFS_train = frFS.fit_transform(richvectors_train, newsgroups_train.target)

print "\ttransforming test set with rich feature set"
richvectors_test = richVectorizer.transform(newsgroups_test.data)
richvectorsFS_test = frFS.transform(richvectors_test)

pickle.dump(richvectors_train, open("20ng_train.p", "wb"))
pickle.dump(richvectors_test, open("20ng_test.p", "wb"))
#richvectors_train = pickle.load(open("20ng_train.p", "rb"))
#richvectors_test = pickle.load(open("20ng_test.p", "rb"))

richclf = MultinomialNB()
richclf.fit(richvectors_train, newsgroups_train.target)

pred = richclf.predict(richvectors_test)
print "Rich Encoding performance is {}".format(metrics.f1_score(newsgroups_test.target, pred, average='macro'))
#print "Rich Encoding performance is {}".format(metrics.f1_score(newsgroups_train.target, richclf.predict(richvectors_train), average='macro'))

#####
# with feature selection using mutual information


