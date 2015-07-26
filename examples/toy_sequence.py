import numpy as np
import sys

np.random.seed(42)
sys.path = ["/home/hannes/checkout/git/Passage"] + sys.path
import toy_sequence_dataset
import theano.tensor as T

from passage.theano_utils import floatX
from passage.models import RNN
from passage.iterators import Padded
from passage.updates import NAG, Regularizer
from passage.layers import Generic, GatedRecurrent, Dense, SimpleRecurrent
from passage.utils import load, save


batch_size = 64
data_dim = 10
n_classes = 4
max_sent_len=10
y_lag = 4
trX, teX, trY, teY = toy_sequence_dataset.generate((1000, 500), data_dim=data_dim,
                                                   n_phrase_labels=n_classes,
                                                   max_sent_len=max_sent_len,
                                                   tag_end=True)

layers = [
	Generic(size=data_dim),
	SimpleRecurrent(size=32, p_drop=0.2, seq_output=True),
	Dense(size=n_classes, activation='softmax', p_drop=0.5)
]

#A bit of l2 helps with generalization, higher momentum helps convergence
updater = NAG(momentum=0.95, regularizer=Regularizer(l2=1e-4))
padder = Padded(size=batch_size, x_dtype=floatX, y_lag=y_lag)

#Linear iterator for real valued data, cce cost for softmax
model = RNN(layers=layers, updater=updater, iterator=padder, cost='cpsce', Y=T.tensor3())
model.fit(trX, trY, n_epochs=100, batch_size=batch_size)

tr_preds = model.predict(trX[:len(teY)])
te_preds = model.predict(teX)

tr_acc = np.mean([np.mean(y == np.argmax(yhat[y_lag:],axis=1)) for y, yhat in zip(trY[:len(teY)], tr_preds)])
te_acc = np.mean([np.mean(y == np.argmax(yhat[y_lag:],axis=1)) for y, yhat in zip(teY, te_preds)])

# Test accuracy should be between 98.9% and 99.3%
print 'train accuracy', tr_acc, 'test accuracy', te_acc
