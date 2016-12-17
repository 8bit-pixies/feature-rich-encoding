from nltk.corpus import reuters 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from FeatureRichEncoding import FeatureRichEncoding
from sklearn.pipeline import FeatureUnion

train_docs = []
test_docs = []

for doc_id in reuters.fileids():
    if doc_id.startswith("train"):
        train_docs.append(reuters.raw(doc_id))
    else:
        test_docs.append(reuters.raw(doc_id))
