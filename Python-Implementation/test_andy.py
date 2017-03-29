from dataset import *

ds_path = '/Users/andymartinez/Desktop/Dataset'

d = Dataset(ds_path)
print str(d)

d.shuffle_dataset()
(X_train, y_train), (X_test, y_test) = d.get_data_split(0.5)
print X_train
print '-----------------------------'
print y_train
print '**********************************'
print X_test
print '-----------------------------'
print y_test
