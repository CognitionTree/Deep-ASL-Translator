from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from dataset import *
from numpy import *
from plotting_tools import *

#Split ratios
train_frac = 0.75
val_frac = 0.05
test_frac = 0.2

#Dataset path  --> Need to modify this
dataset_path = '/Users/danielaflorit/Github/ASL_Dataset/Optical_flow'

#Getting train, validation and test data
dataset = Dataset(dataset_path)
((X_train, y_train),(X_val, y_val),(X_test, y_test)) = dataset.get_data_split(train_frac, val_frac, test_frac)

#Model parameters
num_classes = dataset.get_numb_classes()
n_epochs = 1
batch_size = 10
pool_size = (2, 2)
kernel_size = (3,3)

#Names:
model_name = 'conv_model.png'
cm_name = 'Confusion Matrix Of Experiment Conv model with Optical Flow'

print(X_train.shape, 'train samples')
print(X_val.shape, 'val samples')
print(X_test.shape, 'test samples')

numb_images, img_rows, img_cols, img_channels  = X_train.shape
input_shape = (img_rows, img_cols, img_channels)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
X_val /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Building Model:
model = Sequential()

model.add(Conv2D(32, kernel_size, padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("============================ Model Parameters ============================")
print("Kernel Size: ", kernel_size)
print("Pooling size: ", pool_size)
print("Number of epochs: ", n_epochs)
print("Batch size: ", batch_size)
print("Number of classes: ", num_classes)

print("============================== Model Summary ==============================")
model.summary()

print("================================ Training: ================================")
#Train model
print("Training:")
model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(X_val, y_val), shuffle=True)

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
#feature_vectors = model.predict(X_test)

#Changing y_test to its label for confussion matrix
y_test_label = []
for y in y_test:
	y_test_label.append(argmax(y))

print("============================ Confusion Matrix ============================")
#Building Confusion Matrix
#print(str(dataset.get_gloss_to_numb()))
build_confusion_matrix(y_test_label, predictions, range(dataset.get_numb_classes()), title=cm_name)
#display_feature_vectors(feature_vectors, y_test_label, predictions)
print("==========================================================================")