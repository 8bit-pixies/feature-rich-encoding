from sklearn.base import TransformerMixin
from gensim.models import Word2Vec
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import numpy as np

# based on "AbstractiveTextSummarizationusing Sequence-to-sequence RNNs and Beyond"

class FeatureRichEncoding(TransformerMixin):   

    def __init__(self, tokenizer=None, mode = 'w2v', **kwargs):
        # this tokenizer approach is extremely expensive
        self.model = None
        self.kwargs = kwargs
        self.kwargs['min_count'] = kwargs.get('min_count', 1)
        self.mode = mode
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = nltk.tokenize.word_tokenize

    def get_pos(self, x):
        return [x1[1] for x1 in nltk.pos_tag(x)]

    def get_ner(self, text):
        # http://stackoverflow.com/questions/24398536/named-entity-recognition-with-regular-expression-nltk
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        continuous_chunk = [""]
        current_chunk = []

        for i in chunked:
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
        return continuous_chunk

    def fit(self, x):
        sentences = [self.tokenizer(x1) for x1 in x]
        # we know word2vec can take updating how should we handle this
        if self.mode == 'ner':
            sentences = [self.get_ner(x1) for x1 in x]
        else:
            sentences = [self.tokenizer(x1) for x1 in x]
            if self.mode == 'pos':
                sentences = [self.get_pos(x1) for x1 in sentences]

        self.model = Word2Vec(**self.kwargs)
        self.model.build_vocab(sentences)
        self.model.train(sentences)
        return self

    def transform(self, x, y=None):
        def makeFeatureVec(words):
            # this is taken from the kaggle tutorial
            # in this case, we shall use the representation which is the average
            # vector representation
            # please note that this does not take into account work ordering.
            featureVec = np.zeros(self.model[self.model.index2word[0]].shape, dtype="float32")
            #
            nwords = 0.
            #
            # Index2word is a list that contains the names of the words in
            # the model's vocabulary. Convert it to a set, for speed
            index2word_set = set(self.model.index2word)
            #
            # Loop over each word in the review and, if it is in the model's
            # vocaublary, add its feature vector to the total
            for word in words:
                if word in index2word_set and not np.any(np.isnan(self.model[word])):
                    nwords = nwords + 1.
                    featureVec = np.add(featureVec, self.model[word])
            #
            # Divide the result by the number of words to get the average
            if nwords != 0:
                #print(featureVec)
                featureVec = np.divide(featureVec,nwords)
            return featureVec
        if self.mode == 'ner':
            sentences = [self.get_ner(x1) for x1 in x]
        else:
            sentences = [self.tokenizer(x1) for x1 in x]
            if self.mode == 'pos':
                sentences = [self.get_pos(x1) for x1 in sentences]
        return np.vstack([makeFeatureVec(x1) for x1 in sentences])

if __name__ == "__main__":
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentences = ["It is not known exactly when the text obtained its current standard form",
                 "it may have been as late as the 1960s. Dr. Richard McClintock, a Latin scholar who was the publications director at College in Virginia",
                 "discovered the source of the passage sometime before 1982 while searching for instances of the Latin word"]

    w2vt = FeatureRichEncoding()
    w2vt.fit(sentences)
    w2vt.transform(sentences)

    a = w2vt.fit_transform(sentences)

    w2v_norm = Pipeline([('w2v', FeatureRichEncoding()), ('normalize', StandardScaler())])
    w2v_norm.fit_transform(sentences)

    w2v_norm = Pipeline([('w2v', FeatureRichEncoding(mode='pos')), ('normalize', StandardScaler())])
    w2v_norm.fit_transform(sentences)

    text_feats = FeatureUnion([('w2v', FeatureRichEncoding()), ('pos', FeatureRichEncoding(mode='pos'))])
    combine_feats = text_feats.fit_transform(sentences)

    text_feats = FeatureUnion([('w2v', FeatureRichEncoding()), ('pos', FeatureRichEncoding(mode='pos')),
                               ('ner', FeatureRichEncoding(mode='ner'))])
    combine_feats = text_feats.fit_transform(sentences)

    feature_rich_all = FeatureUnion([('w2v', FeatureRichEncoding()), ('pos', FeatureRichEncoding(mode='pos')),
                               ('ner', FeatureRichEncoding(mode='ner')),
                               ('tfidf', TfidfVectorizer())])
    combine_feats = feature_rich_all.fit_transform(sentences)
