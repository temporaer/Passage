import numpy as np
import sys

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
trX, teX, trY, teY = toy_sequence_dataset.generate((1000, 500), data_dim=data_dim,
                                                   n_phrase_labels=n_classes,
                                                   max_sent_len=max_sent_len)
# trY = [y[-1] for y in trY]
# teY = [y[-1] for y in teY]

Y_ = T.tensor3()
# Y_.tag.test_value = np.random.uniform(size=(batch_size,max_sent_len,n_classes))
X_ = T.tensor3()
# X_.tag.test_value = np.random.uniform(size=(data_dim,batch_size,max_sent_len))
layers = [
	Generic(size=data_dim, input=X_),
	SimpleRecurrent(size=15, p_drop=0.2, seq_output=True),
	Dense(size=n_classes, activation='softmax', p_drop=0.5)
]

#A bit of l2 helps with generalization, higher momentum helps convergence
updater = NAG(momentum=0.95, regularizer=Regularizer(l2=1e-4))
padder = Padded(size=batch_size, x_dtype=floatX)

#Linear iterator for real valued data, cce cost for softmax
model = RNN(layers=layers, updater=updater, iterator=padder, cost='cpsce', Y=Y_)
model.fit(trX, trY, n_epochs=100, batch_size=batch_size)

tr_preds = model.predict(trX[:len(teY)])
te_preds = model.predict(teX)

from IPython.core.debugger import Tracer; Tracer()()
tr_acc = np.mean([np.mean(y == np.argmax(yhat,axis=1)) for y, yhat in zip(trY[:len(teY)], tr_preds)])
te_acc = np.mean([np.mean(y == np.argmax(yhat,axis=1)) for y, yhat in zip(teY, te_preds)])

# Test accuracy should be between 98.9% and 99.3%
print 'train accuracy', tr_acc, 'test accuracy', te_acc
