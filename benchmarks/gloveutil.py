# pre-trained GloVe embeddings...

# ```
# %%bash
# # download GloVe word vector representations
# # bunch of small embeddings - trained on 6B tokens - 822 MB download, 2GB unzipped
# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip
# 
# # and a single behemoth - trained on 840B tokens - 2GB compressed, 5GB unzipped
# wget http://nlp.stanford.edu/data/glove.840B.300d.zip
# unzip glove.840B.300d.zip
# ```
import numpy as np

GLOVE_6B_50D_PATH = "glove.6B.50d.txt"
GLOVE_840B_300D_PATH = "glove.840B.300d.txt"

with open(GLOVE_6B_50D_PATH, "rb") as lines:
    word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}


glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0]
        nums = map(float, parts[1:])
        if word in all_words:
            glove_small[word] = np.array(nums)
            
glove_big = {}
with open(GLOVE_840B_300D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0]
        nums = map(float, parts[1:])
        if word in all_words:
            glove_big[word] = np.array(nums)

# train word2vec on all the texts - both training and test set
# we're not using test labels, just texts so this is fine
"""
from gensim.models.word2vec import Word2Vec
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
model.index2word
w2v = {w: vec for w, vec in zip(model.index2word, model.syn0)}
"""
# Now the meat - classifiers using vector embeddings. We will implement an embedding vectorizer - a counterpart of CountVectorizer and TfidfVectorizer - that is given a word -> vector mapping and vectorizes texts by taking the mean of all the vectors corresponding to individual words.

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())
    
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
# and a tf-idf version of the same

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
            
"""
usage:
```
MeanEmbeddingVectorizer(w2v)
TfidfEmbeddingVectorizer(w2v)
...
MeanEmbeddingVectorizer(glove_small)
```

"""