# Import standard pre-processing libraries for feture engineering
from pandas import read_csv, DataFrame
from numpy.random import seed
import numpy as np

# SciKit Learn Libraries for util purposes
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

# Keras models for building the Neural Net
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K

# Tensorboard to visualise the output
from keras.callbacks import TensorBoard

'''
Step 1: Pre Process the required data in the normal format
'''
# Read the data from the csv file
df = read_csv("kddcup.data", index_col=None)

# Convert the strings to numeric values
df1 = df.replace({'protocol_type': {'tcp':1, 'udp':2, 'icmp':3}})

df2 = df1.replace({'flag': {'SF':1, 'S1':2, 'REJ':3, 'S2':4, 'S0':5, 'S3':6, 'RSTO':7, 'RSTR':8, 'RSTOS0':9, 'OTH':10, 'SH':11}})

df3 = df2.replace({'service': {'http':1, 'smtp':2, 'finger':3, 'domain_u':4, 'auth':5, 'telnet':6, 'ftp':7,
       'eco_i':8, 'ntp_u':9, 'ecr_i':10, 'other':11, 'private':12, 'pop_3':13, 'ftp_data':14,
       'rje':15, 'time':16, 'mtp':17, 'link':18, 'remote_job':19, 'gopher':20, 'ssh':21,
       'name':22, 'whois':23, 'domain':24, 'login':25, 'imap4':26, 'daytime':27, 'ctf':28,
       'nntp':29, 'shell':30, 'IRC':31, 'nnsp':32, 'http_443':33, 'exec':34, 'printer':35,
       'efs':36, 'courier':37, 'uucp':38, 'klogin':39, 'kshell':40, 'echo':41, 'discard':42,
       'systat':43, 'supdup':44, 'iso_tsap':45, 'hostnames':46, 'csnet_ns':47, 'pop_2':48,
       'sunrpc':49, 'uucp_path':50, 'netbios_ns':51, 'netbios_ssn':52, 'netbios_dgm':53,
       'sql_net':54, 'vmnet':55, 'bgp':56, 'Z39_50':57, 'ldap':58, 'netstat':59, 'urh_i':60,
       'X11':61, 'urp_i':62, 'pm_dump':63, 'tftp_u':64, 'tim_i':65, 'red_i':66}})

df = df3.replace({'result': {'normal.':1, 'buffer_overflow.':2, 'loadmodule.':3, 'perl.':4, 'neptune.':5, 'smurf.':6, 'guess_passwd.':7, 'pod.':8, 'teardrop.':9, 'portsweep.':10,
     'ipsweep.':11, 'land.':12, 'ftp_write.':13, 'back.':14, 'imap.':15, 'satan.':16,
     'phf.':17, 'nmap.':18, 'multihop.':19, 'warezmaster.':20, 'warezclient.':21,
     'spy.':22, 'rootkit.':23}})

# Obtain value of X and Y
X = df.iloc[:, 1:-1]
Y = df.result

# Scale value of x between 0 and 1 for each feature
sX = minmax_scale(X, axis = 0)
ncol = sX.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(sX, Y, train_size = 0.7, random_state = seed(2017))

'''
Step 2: Building the Autoencoder
'''

# Get the input dimension of the auto encoder
input_dim = Input(shape = (ncol, ))

# Final encoding dimension that is taken
encoding_dim = 20

# Encoder layers as defined in the paper.
# The last layer has to be taken based on personal choice

encoded1 = Dense(300, activation = 'relu')(input_dim)
encoded2 = Dense(150, activation = 'relu')(encoded1)
encoded3 = Dense(75, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)

# Decoder Layers as required by the encoder layers
# The last layer gives back the output to us
decoded1 = Dense(75, activation = 'relu')(encoded4)
decoded2 = Dense(150, activation = 'relu')(decoded1)
decoded3 = Dense(300, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)

# Combination of encoder and decoder allows us to give the autoencoder
autoencoder = Model(input = input_dim, output = decoded4)

# Simple command to compile and run the autoencoder
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=5, batch_size=100, shuffle=True, validation_data=(X_test, X_test), callbacks=[TensorBoard(log_dir='/tmp/ias-project')])

# Get the encoded input with the reduced dimensionality
encoder = Model(input=input_dim, output=encoded4)
encoded_input = Input(shape=(encoding_dim, ))
encoded_out = encoder.predict(X_test)
print(encoded_out[0:2])

np.savetxt('out.csv', encoded_out, delimiter=',')

from sklearn.svm import SVC, LinearSVC, NuSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from dbn.tensorflow import UnsupervisedDBN, SupervisedDBNClassification

# svm = SVC()
classifier = SupervisedDBNClassification(hidden_layers_structure=[32, 16],
                      batch_size=10,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=2,
                      activation_function='sigmoid')

# classifier = Pipeline(steps=[('dbn', dbn), ('svm', svm)])

X_train, X_test, Y_train, Y_test = train_test_split(encoded_out, Y_test, train_size = 0.7, random_state = seed(2017))

classifier.fit(X_train, Y_train)

print(classification_report(Y_test, classifier.predict(X_test)))
