from models import *
import sys

of_path = 'dataset/Optical_flow'
pair_of_path = '/home/andy/Datasets/ASL/Pair_Optical_flow'


def experiment_conv():
    model_name = 'Convolutional Model Top 4'
    run_conv_model(model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size=(3, 3), pool_size=(2, 2),
                   n_epochs=100, batch_size=10, dataset_path=of_path)


def experiment_lstm(mode, numb_groups):
    model_name = 'LSTM Model AVG Value No Dense' + str(mode) + ' ' + str(numb_groups)
    run_lstm_model(model_name=model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size=(3, 3),
                   pool_size=(2, 2), n_epochs=100, batch_size=10, dataset_path=pair_of_path, mode=mode,
                   numb_groups=numb_groups)


def experiment_bias(model_name='Bias Model'):
    run_bias_model(model_name=model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, dataset_path=of_path)


'''
def experiment_temp_conv():
    model_name = 'Temp Convolutional Model 5 Groups'
    run_temp_conv_model(model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2, kernel_size=(3, 3), pool_size=(2, 2),
                        n_epochs=100, batch_size=10, dataset_path=pair_of_path)
'''


def experiment_random_agent(model_name='Random Model'):
    run_random_model(model_name=model_name, train_frac=0.75, val_frac=0.05, test_frac=0.2,
                     dataset_path=of_path)


# Main
agents = ['random', 'bias', 'conv']

if len(sys.argv) != 2:
    print 'Incorrect number of parameters'
    print 'Run python experiments.py agent'
    print 'Choose any of the tree agents ' + str(agents)
    sys.exit()

chosen_agent = sys.argv[1]

if chosen_agent == agents[0]:
    experiment_random_agent()
elif chosen_agent == agents[1]:
    experiment_bias()
elif chosen_agent == agents[2]:
    experiment_conv()
else:
    print 'Unknown Agent'
    print 'Run python experiments.py agent'
    print 'Choose any of the tree agents ' + str(agents)


# experiment_bias()
# experiment_conv()

# experiment_lstm(None, 3)
# experiment_lstm('max', 5)
# experiment_lstm('avg', 5)

# experiment_lstm(None, 10)
# experiment_lstm('max', 10)
# experiment_lstm('avg', 10)

# experiment_lstm(None, 15)
# experiment_lstm('max', 15)
# experiment_lstm('avg', 15)

# experiment_temp_conv()
# experiment_random_agent()
