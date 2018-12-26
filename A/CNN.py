from keras.models import Sequential
from keras.layers import Dense, Flatten
import keras
from keras.utils import np_utils


class CNN:

    n_classes = 10

    def __init__(self, x_train, y_train, x_test, y_test):
        self.model = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.reshapeAndNormalize()
        self.build()


    def build(self):
        # Build neural
        # Layer             Volume   size  Description
        # INPUT             28x28x1 28x28 pixels and 1 color channel
        # CONV5-32 + ReLU   28x28x32 Conv layer with 32 5x5x1 filters
        # POOL2             14x14x32 Standard 2x2 pooling layer with stride 2
        # CONV5-64 + ReLU   14x14x64 Conv layer with 64 5x5x32 filters
        # POOL2             7x7x64 Standard 2x2 pooling layer with stride 2
        # FC                1024 Fully-connected layer with 1024 units
        # FC                10 Output layer with 10 possible categories

        self.model = Sequential();

        self.model.add(keras.layers.Conv2D(32, kernel_size=[5, 5], strides=[1, 1],
                                      activation='relu',
                                      input_shape=(28, 28, 1)))

        self.model.add(keras.layers.MaxPooling2D(pool_size=[2, 2], strides=(2, 2)))

        self.model.add(keras.layers.Conv2D(64, kernel_size=[5, 5], strides=[1, 1],
                                      activation='relu'))

        self.model.add(keras.layers.MaxPooling2D(pool_size=[2, 2], strides=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(activation="relu", units=1024))

        self.model.add(Dense(activation="softmax", units=self.n_classes))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.01),
                      metrics=['accuracy'])

        return

    def reshapeAndNormalize(self):

        # CNN 4D Tensor from 28x28 pixels
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        # Normalize the CNN data
        self.x_train /= 255
        self.x_test /= 255

        # One-hot training and test data for y
        self.y_train = np_utils.to_categorical(self.y_train, self.n_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.n_classes)


    def train(self):
        self.history = self.model.fit(self.x_train, self.y_train,
                            batch_size=128, epochs=10,
                            verbose=2,
                            validation_data=(self.x_test, self.y_test))
        return

