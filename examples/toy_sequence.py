import numpy as np
import sys

np.random.seed(42)
sys.path = ["/home/hannes/checkout/git/Passage"] + sys.path
import toy_sequence_dataset
import theano.tensor as T

from passage.theano_utils import floatX
from passage.models import RNN
from passage.iterators import Padded
from passage.updates import NAG, Regularizer, Adam
from passage.layers import Generic, GatedRecurrent, Dense, SimpleRecurrent
from passage.utils import load, save


batch_size = 64
data_dim = 10
n_classes = 4
max_sent_len=15
y_lag = 4
trX, teX, trY, teY = toy_sequence_dataset.generate((1000, 500), data_dim=data_dim,
                                                   n_phrase_labels=n_classes,
                                                   max_sent_len=max_sent_len,
                                                   min_sent_len=max_sent_len-3,
                                                   n_phrase_words=4,
                                                   n_phrases=20,
                                                   n_words=15,
                                                   tag_end=False, label_noise=0.0)

layers = [
	Generic(size=data_dim),
	GatedRecurrent(size=32, p_drop=0.1, seq_output=True),
	GatedRecurrent(size=32, p_drop=0.1, seq_output=True),
	Dense(size=n_classes, activation='softmax', p_drop=0.5)
]

#A bit of l2 helps with generalization, higher momentum helps convergence
# updater = NAG(momentum=0.95, regularizer=Regularizer(l2=1e-4))
updater = Adam(lr=0.006)
padder = Padded(size=batch_size, x_dtype=floatX, y_lag=y_lag)

#Linear iterator for real valued data, cce cost for softmax
model = RNN(layers=layers, updater=updater, iterator=padder, cost='cpsce', Y=T.tensor3())
model.fit(trX, trY, n_epochs=100, batch_size=batch_size)

tr_preds = model.predict(trX[:len(teY)])
te_preds = model.predict(teX)

tr_acc = np.mean([np.all(y == np.argmax(yhat[y_lag:],axis=1)) for y, yhat in zip(trY[:len(teY)], tr_preds)])
te_acc = np.mean([np.all(y == np.argmax(yhat[y_lag:],axis=1)) for y, yhat in zip(teY, te_preds)])


print "Examples from Test: "
S = (["\n".join(map(str, (y, np.argmax(yhat[y_lag:],axis=1))))
      for y, yhat in zip(teY, te_preds)])
for s in S[:15]:
    print s, "\n"

# Test accuracy should be between 98.9% and 99.3%
print 'train accuracy', tr_acc, 'test accuracy', te_acc
