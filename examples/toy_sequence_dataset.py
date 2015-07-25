import numpy as np
from numpy.random import uniform, permutation, randint

def contains_any_phrase(sent, phrases):
    """
    return True iff any phrase in phrases is contained in sentence sent
    """
    for p in phrases:
        if p in sent:
            return True
    return False

def generate(size, data_dim=5, n_phrase_labels=4, n_words=3,
             n_phrase_words=3, n_phrases=5, label_noise=0.,
             min_sent_len=5, max_sent_len=5, tag_end=True):
    """
    generate a simple toy dataset for sequence labeling.

    :param size: 2-tuple, number of sequences to generate in train and test
    :param data_dim: input dimensionality (one random vector for each 'word'). 
                    If data_dim=1, return one int per word instead.
    :param n_phrase_labels: number of phrase labens (only end of phrase is labeled)
    :param n_words: number of words in the dictionary
    :param seq_len: number of 'sentences' to generate
    :param n_phrase_words: length of a unique word sequence
    :param n_phrases: unique word sequences within sentences
    :param label_noise: fraction of randomly substituted labels in training set
    :param tag_end: if True label end of phrase, else start
    """
    assert n_words < 256
    assert max_sent_len >= n_phrase_words
    global dictionary, phrases

    # generate dictionary
    dictionary = uniform(size=(n_words, data_dim))

    # generate n_phrases unique word sequences of length n_phrase_words
    phrases = []
    phrase_labels = []
    while len(phrases) != n_phrases:
        phrases = np.unique(np.array(["".join(map(chr, randint(n_words, size=n_phrase_words)))
                             for i in xrange(n_phrases)], dtype=np.object))
        assert np.unique(map(len, phrases)) == n_phrase_words
        phrase_labels = 1+randint(n_phrase_labels-1, size=n_phrases)

    # generate 'sentences'
    Xind = []
    Y = []
    for i in xrange(sum(size)):
        while True:
            sent_len = randint(min_sent_len, max_sent_len+1)
            sent = "".join(map(chr, randint(n_words, size=sent_len)))
            if contains_any_phrase(sent, phrases):
                break
        Y.append(np.zeros(sent_len,dtype=np.int))
        Xind.append(sent)

    # generate labels for dataset
    for phrase, plabel in zip(phrases, phrase_labels):
        for idx, sent in enumerate(Xind):
            start = 0
            while True:
                sidx = sent.find(phrase, start)
                if sidx < 0:
                    break
                if tag_end:
                    Y[idx][sidx+len(phrase)-1] = plabel
                else:
                    Y[idx][sidx] = plabel
                start += 1

    # transform dataset to code
    if data_dim > 1:
        X = [[dictionary[ord(c)] for c in sent] for sent in Xind]
    else:
        X = [[ord(c) for c in sent] for sent in Xind]

    Xtrain, Xtest = X[:size[0]], X[size[0]:]
    Ytrain, Ytest = Y[:size[0]], Y[size[0]:]

    # training label noise
    for sent in Ytrain:
        mask = uniform(0., label_noise, size=sent.size) > 0.
        sent[mask] = randint(n_phrase_labels, size=mask.sum())

    return Xtrain, Xtest, Ytrain, Ytest


if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = generate((5,5), data_dim=1, tag_end=False)
    print "Phrases: ", [",".join(map(str,map(ord,p))) for p in phrases]
    print "Xtrain : ", [",".join(map(str,p)) for p in Xtrain]
    print "Ytrain : ", [",".join(map(str,p)) for p in Ytrain]
