from models import *

of_path = '/home/andy/Datasets/ASL/Optical_flow'
pair_of_path = '/home/andy/Datasets/ASL/Pair_Optical_flow'

def experiment_conv():
	model_name = 'Convolutional Model Top 4'
	run_conv_model(model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size = (3,3), pool_size = (2, 2), n_epochs = 100, batch_size = 10, dataset_path=of_path)

def experiment_lstm(mode, numb_groups):
	model_name = 'LSTM Model AVG Value No Dense' + str(mode) + ' ' + str(numb_groups)
	run_lstm_model(model_name=model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size = (3,3), pool_size = (2, 2), n_epochs = 100, batch_size = 10, dataset_path=pair_of_path, mode=mode, numb_groups=numb_groups)

def experiment_bias(model_name='Bias Model'):
	run_bias_model(model_name=model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, dataset_path=of_path)


def experiment_temp_conv():
	model_name = 'Temp Convolutional Model 5 Groups'
	run_temp_conv_model(model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size = (3,3), pool_size = (2, 2), n_epochs = 100, batch_size = 10, dataset_path=pair_of_path)

#Main
#experiment_bias()
#experiment_conv()

#experiment_lstm(None, 3)
#experiment_lstm('max', 5)
#experiment_lstm('avg', 5)

#experiment_lstm(None, 10)
#experiment_lstm('max', 10)
#experiment_lstm('avg', 10)

#experiment_lstm(None, 15)
#experiment_lstm('max', 15)
#experiment_lstm('avg', 15)

experiment_temp_conv()

