from __future__ import print_function
from numpy import *
from dataset import *

#Model parameters
kernel_size = (5,5)

ds_path = '/Users/danielaflorit/Github/ASL_Dataset/Testing'

d = Dataset(ds_path)
#print str(d)

d.shuffle_dataset()
(X_train, y_train), (X_test, y_test) = d.get_data_split(0.5)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape)

X_train /= 255.0
X_test /= 255.0
