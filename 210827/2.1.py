#케라스에서 Mnist 데이터셋 적재하기
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))


network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)