from imports import *
from util import *
from HybridModel import *

encoding_dim = 20

X_train, X_test, Y_train, Y_test, ncol = pre_process("data/kddcup-test.data")

autoencoder = Autoencoder(encoding_dim, ncol, X_train[0:1000], X_test[0:1000])
encoded_out = autoencoder.train()

deep_belief_network = DBN(encoded_out[0:1000], Y_test[0:1000])
deep_belief_network.train()