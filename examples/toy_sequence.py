import numpy as np
import sys

sys.path = ["/home/hannes/checkout/git/Passage"] + sys.path
import toy_sequence_dataset

from passage.models import RNN
from passage.updates import NAG, Regularizer
from passage.layers import Generic, GatedRecurrent, Dense
from passage.utils import load, save


data_dim = 10
n_labels = 4
trX, teX, trY, teY = toy_sequence_dataset.generate((1000, 500), data_dim=data_dim,
                                                   n_phrase_labels=n_labels)
trY = [y[-1] for y in trY]
teY = [y[-1] for y in teY]

layers = [
	Generic(size=data_dim),
	GatedRecurrent(size=15, p_drop=0.2, seq_output=False),
	Dense(size=n_labels, activation='softmax', p_drop=0.5)
]

#A bit of l2 helps with generalization, higher momentum helps convergence
updater = NAG(momentum=0.95, regularizer=Regularizer(l2=1e-4))

#Linear iterator for real valued data, cce cost for softmax
model = RNN(layers=layers, updater=updater, iterator='linear', cost='cce')
model.fit(trX, trY, n_epochs=200)

tr_preds = model.predict(trX[:len(teY)])
te_preds = model.predict(teX)

tr_acc = np.mean(trY[:len(teY)] == np.argmax(tr_preds, axis=1))
te_acc = np.mean(teY == np.argmax(te_preds, axis=1))

# Test accuracy should be between 98.9% and 99.3%
print 'train accuracy', tr_acc, 'test accuracy', te_acc
