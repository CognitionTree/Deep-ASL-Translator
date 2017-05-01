from models import *

def experiment_conv():
	model_name = 'Convolutional Model Top 4'
	run_conv_model(model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size = (3,3), pool_size = (2, 2), n_epochs = 100, batch_size = 10, dataset_path='/home/andy/Datasets/ASL/Optical_flow')

def experiment_lstm(mode, numb_groups):
	model_name = 'LSTM Model ' + str(mode) + ' ' + str(numb_groups)
	run_lstm_model(model_name=model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size = (3,3), pool_size = (2, 2), n_epochs = 100, batch_size = 10, dataset_path='/home/andy/Datasets/ASL/Pair_Optical_flow', mode=mode, numb_groups=numb_groups)

#Main
experiment_conv()

#experiment_lstm(None, 5)
#experiment_lstm('max', 5)
#experiment_lstm('avg', 5)

#experiment_lstm(None, 10)
#experiment_lstm('max', 10)
#experiment_lstm('avg', 10)

#experiment_lstm(None, 15)
#experiment_lstm('max', 15)
#experiment_lstm('avg', 15)

