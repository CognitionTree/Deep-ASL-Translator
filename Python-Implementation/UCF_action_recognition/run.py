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

#Add the path to utils to be able to import all util files:
import sys
sys.path.append('/Users/danielaflorit/Github/Deep_ASL_Translator/Python-Implementation/tools/utils')
from plotting_utils import *

#Model parameters
kernel_size = (5,5)
input_shape_conv = (0,0,0)
input_shape_time_dist = (0,0,0,0)
batch_size = 10
n_epochs = 50

#Names:
cm_name = 'Confusion Matrix Of Experiment LSTM model with UCF sport dataset' 
model_name = 'model_lstm.png'

#Dataset path  --> Need to modify this
ds_path = '/Users/danielaflorit/Github/UCF_sports_action/Frames'

d = Dataset(ds_path)
d.shuffle_dataset()

#Getting train and test data
(X_train, y_train), (X_test, y_test) = d.get_data_split(0.8)
(numb_examples, numb_frames, rows, cols) = X_train.shape

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

if K.image_data_format() == 'channels_first':
	X_train = X_train.reshape(numb_examples, numb_frames, 1, rows, cols)
	X_test = X_test.reshape(X_test.shape[0], numb_frames, 1, rows, cols)
	input_shape_conv = (1, rows, cols)
	input_shape_time_dist = (numb_frames, 1, rows, cols)
else:
	X_train = X_train.reshape(numb_examples, numb_frames, rows, cols, 1)
	X_test = X_test.reshape(X_test.shape[0], numb_frames, rows, cols, 1)
	input_shape_conv = (rows, cols, 1)
	input_shape_time_dist = (numb_frames, rows, cols, 1)

#Converting y to categorical
y_train = keras.utils.to_categorical(y_train, d.get_numb_classes())
y_test = keras.utils.to_categorical(y_test, d.get_numb_classes())

#Building Model:
model = Sequential()
model.add(TimeDistributed(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', input_shape=(rows,cols, 1)), input_shape=input_shape_time_dist))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(128)))
model.add(TimeDistributed(Dropout(0.5)))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(d.get_numb_classes(), activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

print("============================ Model Parameters ============================")
print("Kernel Size: ", kernel_size)
print("Number of epochs: ", n_epochs)
print("Batch size: ", batch_size)

print("============================== Model Summary ==============================")
model.summary()

print("================================ Training: ================================")
#Train model
model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs,verbose=1, validation_split=0.1)


#Test model
print("=========================== Evaluating Model: ===========================")
score = model.evaluate(X_test, y_test, verbose=1)

print("=========================== Evaluation Metrics ===========================")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Plotting model
#plot_model(model, to_file=model_name, show_shapes=True, show_layer_names=True)

print("=========================== Predicting Labels: ===========================")
#Checking predictions
predictions = model.predict_classes(X_test)

#Changing y_test to its label for confussion matrix
y_test_label = []
for y in y_test:
	y_test_label.append(argmax(y))

#Building Confusion Matrix
plot_confusion_matrix(y_test_label, predictions, range(d.get_numb_classes()), title=cm_name)

