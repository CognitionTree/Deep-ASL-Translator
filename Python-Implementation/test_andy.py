from __future__ import print_function
from numpy import *
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed
from keras import backend as K
from keras.utils import plot_model
from keras import metrics
from dataset import *

#Model parameters
kernel_size = (5,5)

ds_path = '/Users/andymartinez/Datasets/ASL/Frames'

d = Dataset(ds_path)
#print str(d)

d.shuffle_dataset()
(X_train, y_train), (X_test, y_test) = d.get_data_split(0.5)
'''print X_train.shape
print '-----------------------------'
print y_train.shape
print '**********************************'
print X_test.shape
print '-----------------------------'
print y_test.shape'''

X_train = X_train.astype('float32')
(numb_examples, numb_frames, rows, cols) = X_train.shape
input_shape_time_dist = (numb_frames, rows, cols)

X_test = X_test.astype('float32')
#print(X_train.shape)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
X_train /= 255.0
X_test /= 255.0

y_train = keras.utils.to_categorical(y_train, d.get_numb_classes())
y_test = keras.utils.to_categorical(y_test, d.get_numb_classes())
#print(X_train.shape)
model = Sequential()
model.add(TimeDistributed(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', input_shape=(rows,cols)), input_shape=input_shape_time_dist))