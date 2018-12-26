from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

class Linear:

    n_classes=10

    def __init__(self, x_train, y_train,x_test, y_test):
        self.model=None
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.history = None
        self.reshapeAndNormalize()
        self.buildLinear()


    def buildLinear(self):

        self.model = Sequential();

        self.model.add(Dense(units=512, activation="relu", input_dim=784))

        self.model.add(Dense(units=512, activation="relu"))

        self.model.add(Dense(units=self.n_classes, activation="softmax"))

        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        return

    def reshapeAndNormalize(self):

        # Reshape training data for use with linear clasifier
        self.x_train = self.x_train.reshape(60000, 784)
        self.x_test = self.x_test.reshape(10000, 784)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        # normalizing linear data
        self.x_train /= 255
        self.x_test /= 255

        #One-hot training and test data for y
        self.y_train = np_utils.to_categorical(self.y_train, self.n_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.n_classes)


    def train(self):

        self.history = self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=15, verbose=2, validation_data=(self.x_test, self.y_test))

