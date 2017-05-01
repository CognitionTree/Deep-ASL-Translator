from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM
from keras.utils import plot_model
from image_dataset import *
from video_dataset import *
from numpy import *
from plotting_tools import *
from keras import regularizers
import matplotlib.pyplot as plt
from scipy.io import *


def save_keras_model(name, model):
	trained_models_dir = 'trained_models/'

	model_json = model.to_json()
	with open(trained_models_dir+ name + '.json', 'w') as json_file:
		json_file.write(model_json)
		
	model.save_weights(trained_models_dir + name + ".h5")

def evaluation(model, history, X_test, y_test, num_classes, model_image, cm_name, acc_history_name, loss_history_name, mat_file, parameters, model_name):
	#Test model
	print("=========================== Evaluating Model: ===========================")
	score = model.evaluate(X_test, y_test, verbose=1)
	
	print("=========================== Evaluation Metrics ===========================")
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	
	#Checking predictions
	print("=========================== Predicting Labels: ===========================")
	predictions = model.predict_classes(X_test)

	#Changing y_test to its label for confussion matrix
	y_test_label = []
	for y in y_test:
		y_test_label.append(argmax(y))

	#Building Confusion Matrix
	print("============================ Confusion Matrix ============================")
	build_confusion_matrix(y_test_label, predictions, range(num_classes), title=cm_name)

	#Save Trained Model And Weights
	save_keras_model(model_name, model)
	plot_model(model, to_file=model_image, show_shapes=True, show_layer_names=True)


	#summarize history for accuracy 
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(acc_history_name)
	plt.close()

	#summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(loss_history_name)
	plt.close()

	#Saving Results and history
	test_accuracy = array([score[1]], dtype=object)
	test_loss = array([score[0]], dtype=object)
	test_labels=array(y_test_label, dtype=object)
	test_predictions=array(predictions, dtype=object)

	history_train_accuracy = array(history.history['acc'], dtype=object)
	history_train_loss = array(history.history['loss'], dtype=object)
	history_val_accuracy = array(history.history['val_acc'], dtype=object)
	history_val_loss = array(history.history['val_loss'], dtype=object)
	
	savemat(mat_file, mdict={'test_acc': test_accuracy, 'test_loss':test_loss, 'test_labels':test_labels, 'test_pred':test_predictions, 'history_train_acc':history_train_accuracy, 'history_train_loss':history_train_loss, 'history_val_loss':history_val_loss, 'history_val_acc':history_train_accuracy, 'parameters':parameters})
	#TODO TSNE







def conv_model(input_shape, num_classes, kernel_size = (3,3), pool_size = (2, 2)):
	model = Sequential()

	model.add(Conv2D(32, kernel_size, padding='same',input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size, padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dropout(0.25))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def run_conv_model(model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size = (3,3), pool_size = (2, 2), n_epochs = 100, batch_size = 10, dataset_path='/home/andy/Datasets/ASL/Optical_flow'):
	model_image = 'model_images/'+model_name+'.png'
	cm_name = model_name + ' Confusion Matrix'
	acc_history_name = 'results/' + model_name + '_accuracy_history.png'
	loss_history_name = 'results/' + model_name + '_loss_history.png'
	mat_file = 'results/' + model_name + '.mat'

	#Getting train, validation and test data
	dataset = Image_Dataset(dataset_path)
	((X_train, y_train),(X_val, y_val),(X_test, y_test)) = dataset.get_data_split(train_frac, val_frac, test_frac)
	
	#Model parameters
	num_classes = dataset.get_numb_classes()

	#Preprocessing
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


	#Building the model
	model = conv_model(input_shape, num_classes, kernel_size, pool_size)

	#Printing Model Information
	print("============================ Model Summary ============================")
	model.summary()

	print("============================ Model Information ============================")
	print('Model Name: ', model_name)
	print('Model Image File Name: ', model_image)
	print('Confusion Matrix Name: ', cm_name)
	print("Kernel Size: ", kernel_size)
	print("Pooling size: ", pool_size)
	print("Number of epochs: ", n_epochs)
	print("Batch size: ", batch_size)
	print("Number of classes: ", num_classes)

	print("============================ Input Shapes ============================")
	print(X_train.shape, ' input train samples')
	print(X_val.shape, ' input val samples')
	print(X_test.shape, ' input test samples')
	print(y_train.shape, ' label train samples')
	print(y_val.shape, ' label val samples')
	print(y_test.shape, ' label test samples')
	

	#Training the model to fit dataset
	print("================================ Training: ================================")
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(X_val, y_val), shuffle=True)
	
	parameters = array([train_frac, val_frac, test_frac, kernel_size[0], pool_size[0], n_epochs, batch_size], dtype=object)
	evaluation(model, history, X_test, y_test, num_classes, model_image, cm_name, acc_history_name, loss_history_name, mat_file, parameters, model_name)

	
	
def lstm_model(input_shape_conv, time_distributed_input_shape, num_classes, kernel_size = (3,3), pool_size = (2, 2), mode=None):
	model = Sequential()

	model.add(TimeDistributed(Conv2D(32, kernel_size, padding='same',input_shape=input_shape_conv), input_shape=time_distributed_input_shape))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(Dropout(0.25)))
	model.add(TimeDistributed(Conv2D(64, kernel_size, padding='same')))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))
	model.add(TimeDistributed(Dropout(0.25)))
	model.add(TimeDistributed(Flatten()))
	model.add(TimeDistributed(Dense(512, kernel_regularizer=regularizers.l2(0.01))))
	model.add(TimeDistributed(Activation('relu')))

	if mode == None:
		model.add(LSTM(512, return_sequences=False))
	elif mode == 'max':
		model.add(LSTM(512, return_sequences=True))
		model.add(GlobalMaxPooling1D())
	else:
		model.add(LSTM(512, return_sequences=True))
		model.add(GlobalAveragePooling1D())	
		
	model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dropout(0.25))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model




def run_lstm_model(model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size = (3,3), pool_size = (2, 2), n_epochs = 100, batch_size = 10, dataset_path='/home/andy/Datasets/ASL/Pair_Optical_flow', mode=None, numb_groups=35):
	model_image = 'model_images/'+model_name+'.png'
	cm_name = model_name + ' Confusion Matrix'
	acc_history_name = 'results/' + model_name + '_accuracy_history.png'
	loss_history_name = 'results/' + model_name + '_loss_history.png'
	mat_file = 'results/' + model_name + '.mat'

	#Getting train, validation and test data
	dataset = Video_Dataset(dataset_path)
	(X_train, y_train), (X_val, y_val),(X_test, y_test) = dataset.get_data_split(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, numb_groups=numb_groups)
	
	#Model parameters
	num_classes = dataset.get_numb_classes()

	#Preprocessing
	numb_videos, numb_images, img_rows, img_cols, img_channels  = X_train.shape
	input_shape_conv = (img_rows, img_cols, img_channels)
	time_distributed_input_shape = (numb_images, img_rows, img_cols, img_channels)
	
	X_train = X_train.astype('float32')
	X_val = X_val.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_val /= 255
	X_test /= 255

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)


	#Building the model
	model = lstm_model(input_shape_conv, time_distributed_input_shape, num_classes, kernel_size =kernel_size, pool_size=pool_size, mode=mode)

	#Printing Model Information
	print("============================ Model Summary ============================")
	model.summary()

	print("============================ Model Information ============================")
	print('Model Name: ', model_name)
	print('Model Image File Name: ', model_image)
	print('Confusion Matrix Name: ', cm_name)
	print("Kernel Size: ", kernel_size)
	print("Pooling size: ", pool_size)
	print("Number of epochs: ", n_epochs)
	print("Batch size: ", batch_size)
	print("Number of classes: ", num_classes)
	print('LSTM Mode: ', mode)
	print('Number Of Groups: ', numb_groups)

	print("============================ Input Shapes ============================")
	print(X_train.shape, ' input train samples')
	print(X_val.shape, ' input val samples')	
	print(X_test.shape, ' input test samples')
	print(y_train.shape, ' label train samples')
	print(y_val.shape, ' label val samples')
	print(y_test.shape, ' label test samples')
	print(input_shape_conv, ' convolutional input shape')
	print(time_distributed_input_shape, ' time distributed input shape')
	
	
	#Training the model to fit dataset
	print("================================ Training: ================================")
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(X_val, y_val), shuffle=True)
	
	parameters = array([train_frac, val_frac, test_frac, kernel_size[0], pool_size[0], n_epochs, batch_size, numb_groups], dtype=object)
	evaluation(model, history, X_test, y_test, num_classes, model_image, cm_name, acc_history_name, loss_history_name, mat_file, parameters, model_name)




