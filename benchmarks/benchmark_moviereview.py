import random
from nltk.corpus import movie_reviews

reviews = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
new_train, new_test = reviews[0:100], reviews[101:200]

train_feats, train_label = zip(*new_train)
test_feats, test_label = zip(*new_test)

train_feats = [' '.join(x) for x in train_feats]
test_feats = [' '.join(x) for x in test_feats]


              

