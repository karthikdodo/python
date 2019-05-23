import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt

#setting the batch size and epochs
from sklearn.model_selection import train_test_split

batch_size = 128
nb_classes = 10
nb_epoch = 30

#loading the dataset
dataset = pd.read_csv("heart.csv", header=None).values
 #print(dataset)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[1:,0:13], dataset[1:,13],
                                                    test_size=0.25, random_state=87)
##Data normalization
X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)
X_train /= 255
X_test /= 255
print(X_train)
Y_Train = np_utils.to_categorical(Y_train, nb_classes)
Y_Test = np_utils.to_categorical(Y_test, nb_classes)

# Linear regression
model = Sequential()
model.add(Dense(output_dim=10, input_shape=(13,), init='normal', activation='relu'))
model.compile(optimizer='rmsprop', loss='mean_absolute_error', metrics=['accuracy'])
model.summary()

#Tensorboard log generation for graphs
tensorboard = TensorBoard(log_dir="1qgraphs-2/",histogram_freq=0, write_graph=True, write_images=True)

#fitting the model
history=model.fit(X_train, Y_Train, nb_epoch=nb_epoch, batch_size=batch_size,callbacks=[tensorboard])

#predicting the accuracy of the model
score = model.evaluate(X_test, Y_Test, verbose=1)
print('Loss: %.2f, Accuracy: %.2f' % (score[0], score[1]))

#plotting the loss
plt.plot(history.history['loss'])
# plt.plot(history.history['test_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()