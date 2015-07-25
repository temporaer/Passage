import numpy as np

from utils import shuffle, iter_data
from theano_utils import floatX, intX
from collections import Iterable

def _padded(seqs, return_sizes=False):
    if False:
        # ret = np.rollaxis(np.asarray(seqs),1,0)
        ret = np.asarray(seqs)
        print "_padded: ", ret.shape
        if return_sizes:
            return ret, np.ones_like(ret)
        return ret
    lens = map(len, seqs)
    max_len = max(lens)
    seqs_padded = []
    padding_sizes = []
    feature0 = seqs[0][0]
    if isinstance(feature0, Iterable):
        feature0 = np.zeros_like(feature0)
    else:
        feature0 = 0
    for seq, seq_len in zip(seqs, lens):
        n_pad = max_len - seq_len 
        if return_sizes:
            p = np.ones(max_len)
            p[:n_pad] = 0
            padding_sizes.append(p)
        if n_pad > 0:
            seq = np.vstack(([feature0] * n_pad, seq))
        seqs_padded.append(seq)

    ret = np.rollaxis(np.asarray(seqs_padded), 1, 0)
    if return_sizes:
        return ret, np.array(padding_sizes)
    return ret

class Linear(object):
    """
    Useful for training on real valued data where first dimension is examples, 
    second dimension is to be iterated over, and third dimension is data vectors.

    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    x_dtype is for casting input data
    y_dtype is for casting target data
    """

    def __init__(self, size=64, shuffle=True, x_dtype=floatX, y_dtype=floatX):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb = self.x_dtype(xmb)
            shape = range(len(xmb.shape))
            shape[0] = 1
            shape[1] = 0
            shape = tuple(shape)
            xmb = xmb.transpose(*shape)
            yield xmb

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            xmb = self.x_dtype(xmb)
            shape = range(len(xmb.shape))
            shape[0] = 1
            shape[1] = 0
            shape = tuple(shape)
            xmb = xmb.transpose(*shape)
            ymb = self.y_dtype(ymb)
            if ymb.ndim == 3:
                # sequence prediction! also reorder ymb.
                ymb = ymb.transpose(*shape)
            yield xmb, ymb

class Padded(object):

    def __init__(self, size=64, shuffle=True, x_dtype=intX, y_dtype=floatX):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

    def iterX(self, X):

        for xmb in iter_data(X, size=self.size):
            xmb, padsizes = _padded(xmb, return_sizes=True)
            yield self.x_dtype(xmb), padsizes.T

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for xmb, ymb in iter_data(X, Y, size=self.size):
            xmb = _padded(xmb)
            if ymb[0].ndim == 2:
                # sequence prediction
                ymb, padsize = _padded(ymb, return_sizes=True)
                yield self.x_dtype(xmb), (self.y_dtype(ymb), padsize.T)
            else:
                yield self.x_dtype(xmb), self.y_dtype(ymb)

class SortedPadded(object):

    def __init__(self, size=64, shuffle=True, x_dtype=intX, y_dtype=floatX):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

    def iterX(self, X):
        for x_chunk, chunk_idxs in iter_data(X, np.arange(len(X)), size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            chunk_idxs = [chunk_idxs[idx] for idx in sort]
            for xmb, idxmb in iter_data(x_chunk, chunk_idxs, size=self.size):
                xmb = _padded(xmb)
                yield self.x_dtype(xmb), idxmb   

    def iterXY(self, X, Y):
        
        if self.shuffle:
            X, Y = shuffle(X, Y)

        for x_chunk, y_chunk in iter_data(X, Y, size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            y_chunk = [y_chunk[idx] for idx in sort]
            mb_chunks = [[x_chunk[idx:idx+self.size], y_chunk[idx:idx+self.size]] for idx in range(len(x_chunk))[::self.size]]
            mb_chunks = shuffle(mb_chunks)
            for xmb, ymb in mb_chunks:
                xmb = _padded(xmb)
                if ymb.ndim == 3:
                    ymb, padsize = _padded(ymb, return_sizes=True)
                    yield self.x_dtype(xmb), (self.y_dtype(ymb), padsize.T)
                else:
                    yield self.x_dtype(xmb), self.y_dtype(ymb)
