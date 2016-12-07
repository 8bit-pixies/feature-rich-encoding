Feature Rich Encoding
=====================

This is a simple Python library which adds the ability to create "feature rich encodings" which were described by Nallapati _et al_(2016) and is built ontop of `scikit-learn` library.

The key idea is to concantenate word embeddings for:

*  Word2Vec
*  POS
*  NER
*  tfidf

Each of word2vec, POS, and NER were converted to a word embedding using word2vec module within Gensim.

Usage
=====

Usage can be viewed from `fre.py`, and can easily be implemented into your `sklearn.Pipeline` workflow:

```py
from FeatureRichEncoding import FeatureRichEncoding
sentences = ["It is not known exactly when the text obtained its current standard form",
             "it may have been as late as the 1960s. Dr. Richard McClintock, a Latin scholar who was the publications director at College in Virginia",
             "discovered the source of the passage sometime before 1982 while searching for instances of the Latin word"]

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

feature_rich_all = FeatureUnion([('w2v', FeatureRichEncoding()), ('pos', FeatureRichEncoding(mode='pos')),
                          ('ner', FeatureRichEncoding(mode='ner')),
                          ('tfidf', TfidfVectorizer())])
combine_feats = feature_rich_all.fit_transform(sentences)
```

Requirments
===========

*  `gensim`
*  `nltk` : you may need to download some of the relevant corpus as well.
*  `scikit-learn`

Installation
============

```
python setup.py install
```

References
==========

Nallapati, R., Xiang, B., & Zhou, B. (2016). Sequence-to-sequence rnns for text summarization. _arXiv preprint arXiv:1602.06023._ Retreived from <https://arxiv.org/abs/1602.06023>
