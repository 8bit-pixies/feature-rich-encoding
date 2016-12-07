from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from FeatureRichEncoding import FeatureRichEncoding
from sklearn.pipeline import FeatureUnion    

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

print "Training with only TFIDF"

vectorizer = TfidfVectorizer()

vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

clf = MultinomialNB(alpha=.01)
clf.fit(vectors_train, newsgroups_train.target)

pred = clf.predict(vectors_test)
print "Base line performance is {}".format(metrics.f1_score(newsgroups_test.target, pred, average='macro'))

print "Training with default Feature Rich Encoding"

richVectorizer = FeatureUnion([('w2v', FeatureRichEncoding(max_vocab_size=10**6, batch_words=10**3)), 
                               ('pos', FeatureRichEncoding(mode='pos', max_vocab_size=10**6, batch_words=10**3)),
                               ('ner', FeatureRichEncoding(mode='ner', max_vocab_size=10**6, batch_words=10**3)),
                               ('tfidf', TfidfVectorizer())])

print "\ttransforming training set with rich feature set"
richvectors_train = richVectorizer.fit_transform(newsgroups_train.data)
print "\ttransforming test set with rich feature set"
richvectors_test = richVectorizer.transform(newsgroups_test.data)

richclf = MultinomialNB(alpha=.01)
richclf.fit(richvectors_train, newsgroups_train.target)

pred = richclf.predict(richvectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')

print "Training with default Feature Rich Encoding and LDA"

# insert code
