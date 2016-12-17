from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif, SelectPercentile, VarianceThreshold

#from FeatureRichEncoding import FeatureRichEncoding
from sklearn.pipeline import FeatureUnion, Pipeline   

#import pickle

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
print "Base line train performance is {}".format(metrics.f1_score(newsgroups_train.target, clf.predict(vectorsFS_train), average='macro'))
print "Base line test performance is {}\n".format(metrics.f1_score(newsgroups_test.target, pred, average='macro'))

