from imports import *

class Autoencoder:
    def __init__(self, ncol):
        self.input_dim = Input(shape=(ncol, ))
        self.encoding_dim = 20

    def train(self):
        self.encoded1 = Dense(300, activation='relu')(input_dim)
        self.encoded2 = Dense(150, activation='relu')(encoded1)
        self.encoded3 = Dense(75, activation='relu')(encoded2)
        self.encoded4 = Dense(encoding_dim, activation='relu')(encoded3)

        # Decoder Layers as required by the encoder layers
        # The last layer gives back the output to us
        self.decoded1 = Dense(75, activation='relu')(encoded4)
        self.decoded2 = Dense(150, activation='relu')(decoded1)
        self.decoded3 = Dense(300, activation='relu')(decoded2)
        self.decoded4 = Dense(ncol, activation='sigmoid')(decoded3)

        # Combination of encoder and decoder allows us to give the autoencoder
        self.autoencoder = Model(input=input_dim, output=decoded4)

        # Simple command to compile and run the autoencoder
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.fit(X_train, X_train, epochs=5, batch_size=100, shuffle=True, validation_data=(
            X_test, X_test), callbacks=[TensorBoard(log_dir='/tmp/ias-project')])

        # Get the encoded input with the reduced dimensionality
        self.encoder = Model(input=input_dim, output=encoded4)
        self.encoded_input = Input(shape=(encoding_dim, ))
        self.encoded_out = encoder.predict(X_test)
        print(encoded_out[0:2])

        np.savetxt('out.csv', encoded_out, delimiter=',')

        return encoded_out

class DBN:
    def __init__(self):
        self.input = 10

    def train(self):
        # svm = SVC()
        classifier = SupervisedDBNClassification(hidden_layers_structure=[32, 16],
                                                batch_size=10,
                                                learning_rate_rbm=0.06,
                                                n_epochs_rbm=2,
                                                activation_function='sigmoid')

        # classifier = Pipeline(steps=[('dbn', dbn), ('svm', svm)])

        X_train, X_test, Y_train, Y_test = train_test_split(
            encoded_out, Y_test, train_size=0.7, random_state=seed(2017))

        classifier.fit(X_train, Y_train)

        Y_pred = classifier.predict(X_test)
        print('Accuracy for Deep Belief Network: %f' % accuracy_score(Y_test, Y_pred))
        print(classification_report(Y_test, classifier.predict(X_test)))

# Get the input dimension of the auto encoder
# input_dim = Input(shape=(ncol, ))

# self.Final encoding dimension that is take# encoding_dim = self.self.self.20

# Encoder layers as defined in the paper.
# The last layer has to be taken based on personal self.self.self.self.choice

# encoded1 = Dense(300, activation='relu')(input_dim)
# encoded2 = Dense(150, activation='relu')(encoded1self.)
# encoded3 = Dense(75, activation='relu')(encoded2)
# encoded4 = Dense(encoding_dim, activation='relu')(encoded3selfself..)

# # Decoder Layers as required by the encoder layers
# # The last layer gives back the output to us
# decoded1 = Dense(75, activation='relu')(encoded4)
# decoded2 = Dense(150, activation='relu')(decoded1)
# decoded3 = Dense(300, activation='relu')(decoded2)
# decoded4 = Dense(ncol, activation='sigmoid')(decoded3)

# # Combination of encoder and decoder allows us to give the autoencoder
# autoencoder = Model(input=input_dim, output=decoded4)

# # Simple command to compile and run the autoencoder
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# autoencoder.fit(X_train, X_train, epochs=5, batch_size=100, shuffle=True, validation_data=(
#     X_test, X_test), callbacks=[TensorBoard(log_dir='/tmp/ias-project')])

# # Get the encoded input with the reduced dimensionality
# encoder = Model(input=input_dim, output=encoded4)
# encoded_input = Input(shape=(encoding_dim, ))
# encoded_out = encoder.predict(X_test)
# print(encoded_out[0:2])

# np.savetxt('out.csv', encoded_out, delimiter=',')

# from sklearn.svm import SVC, LinearSVC, NuSVR
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report
# from dbn.tensorflow import UnsupervisedDBN, SupervisedDBNClassification
