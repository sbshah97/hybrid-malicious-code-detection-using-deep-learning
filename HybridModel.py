from imports import *

class Autoencoder:
    def __init__(self, encoding_dim, ncol, X_train, X_test):
        self.input_dim = Input(shape=(ncol, ))
        self.encoding_dim = encoding_dim
        self.X_train = X_train
        self.X_test = X_test
        self.ncol = ncol

    def train(self):
        self.encoded1 = Dense(300, activation='relu')(self.input_dim)
        self.encoded2 = Dense(150, activation='relu')(self.encoded1)
        self.encoded3 = Dense(75, activation='relu')(self.encoded2)
        self.encoded4 = Dense(self.encoding_dim, activation='relu')(self.encoded3)

        # Decoder Layers as required by the encoder layers
        # The last layer gives back the output to us
        self.decoded1 = Dense(75, activation='relu')(self.encoded4)
        self.decoded2 = Dense(150, activation='relu')(self.decoded1)
        self.decoded3 = Dense(300, activation='relu')(self.decoded2)
        self.decoded4 = Dense(self.ncol, activation='sigmoid')(self.decoded3)

        # Combination of encoder and decoder allows us to give the autoencoder
        self.autoencoder = Model(input=self.input_dim, output=self.decoded4)

        # Simple command to compile and run the autoencoder
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.fit(self.X_train, self.X_train, epochs=5, batch_size=100, shuffle=True, validation_data=(
            self.X_test, self.X_test), callbacks=[TensorBoard(log_dir='/tmp/ias-project')])

        # Get the encoded input with the reduced dimensionality
        self.encoder = Model(input=self.input_dim, output=self.encoded4)
        self.encoded_input = Input(shape=(self.encoding_dim, ))
        encoded_out = self.encoder.predict(self.X_test)
  
        np.savetxt('out.csv', encoded_out, delimiter=',')

        return encoded_out

class DBN:
    def __init__(self, X, Y):
        self.input = 10
        self.X = X
        self.Y = Y

    def train(self):
        # svm = SVC()
        self.classifier = SupervisedDBNClassification(hidden_layers_structure=[32, 16],
                                                batch_size=10,
                                                learning_rate_rbm=0.06,
                                                n_epochs_rbm=2,
                                                activation_function='sigmoid')

        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, train_size=0.7, random_state=seed(2017))

        print(X_train[:2], Y_train[:2])

        self.classifier.fit(X_train[0], Y_train)

        Y_pred = self.classifier.predict(X_test)
        print('Accuracy for Deep Belief Network: %f' % accuracy_score(Y_test, Y_pred))
        print(classification_report(Y_test, self.classifier.predict(X_test)))