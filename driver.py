from imports import *
from util import *
from HybridModel import *

encoding_dim = 20

X, Y, ncol = pre_process("data/kddcup-test.data")

autoencoder = Autoencoder(encoding_dim, ncol, X)
encoded_out = autoencoder.train()

deep_belief_network = DBN(X, Y)
deep_belief_network.train()