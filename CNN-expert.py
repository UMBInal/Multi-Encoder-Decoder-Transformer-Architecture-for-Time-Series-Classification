## Written by Mr. Inal Mashukov
## Simple CNN model, for code testing
from tensorflow.keras import layers, Model
import keras

class CNN(Model):

  def __init__(self, n_neurons, num_classes, dropout):
    super(CNN, self).__init__()
    self.n_neurons = n_neurons
    self.num_classes = num_classes
    self.dropout = dropout

    self.conv1 = keras.layers.Conv1D(filters = 32, kernel_size = 2, use_bias = True, activation = 'relu')
    self.maxpool1 = keras.layers.MaxPool1D(pool_size=2)
    self.dense1 = keras.layers.Dense(n_neurons, activation = 'relu')
    self.dropout1 = keras.layers.Dropout(dropout)
    self.dense2 = keras.layers.Dense(n_neurons, activation = 'relu')
    self.dropout2 = keras.layers.Dropout(dropout)
    self.softmax = keras.layers.Dense(num_classes, activation = 'softmax')

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.maxpool1(x)
    x = tf.keras.layers.Flatten()(x)
    x = self.dense1(x)
    x = self.dropout1(x)
    x = self.dense2(x)
    x = self.dropout2(x)
    x = self.softmax(x)
    # print(x.shape)
    # print('3'*100)
    print(f'cnn shape: {x.shape}')
    return x
