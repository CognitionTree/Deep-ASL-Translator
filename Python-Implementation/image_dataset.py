import glob
from frame import *
from random import shuffle
from numpy import *


class Image_Dataset(object):
    def __init__(self, path, view='Front'):
        self.path = path
        self.view = view
        self.signs = []
        self.gloss_count = {}
        self.gloss_to_numb = {}

        self.read_signs()

    def read_signs(self):
        all_sign_paths = glob.glob(self.path + '/*')

        # Including the view in the path
        for i in range(len(all_sign_paths)):
            all_sign_paths[i] = all_sign_paths[i] + '/' + self.view

        numb = 0
        for sign_path in all_sign_paths:
            imgs_paths = glob.glob(sign_path + '/*.png')

            for img_path in imgs_paths:
                sign = Frame(img_path)
                self.signs.append(sign)

                gloss = sign.get_gloss()
                if gloss in self.gloss_count:
                    self.gloss_count[gloss] += 1.0
                else:
                    self.gloss_count[gloss] = 1.0
                    self.gloss_to_numb[gloss] = numb
                    numb += 1

        shuffle(self.signs)

    def get_gloss_to_numb(self):
        return self.gloss_to_numb

    def get_numb_classes(self):
        return len(self.gloss_count)

    def compute_zero_mean(self, I):
        I = I.astype('float32')
        I = I / 255.0
        mean = I.mean(axis=0)
        I_zero_mean = I - mean
        return I_zero_mean

    # train + val + test = 1
    def get_data_split(self, train_frac=0.75, val_frac=0.05, test_frac=0.2):
        train_gloss_count = dict.fromkeys(self.gloss_count, 0.0)
        val_gloss_count = dict.fromkeys(self.gloss_count, 0.0)
        test_gloss_count = dict.fromkeys(self.gloss_count, 0.0)

        X_train = []
        y_train = []

        X_val = []
        y_val = []

        X_test = []
        y_test = []

        for sign in self.signs:
            img = sign.get_img()
            gloss = sign.get_gloss()

            if train_gloss_count[gloss] / self.gloss_count[gloss] < train_frac:
                X_train.append(img)
                y_train.append(self.gloss_to_numb[gloss])
                train_gloss_count[gloss] += 1
            elif test_gloss_count[gloss] / self.gloss_count[gloss] < test_frac:
                X_test.append(img)
                y_test.append(self.gloss_to_numb[gloss])
                test_gloss_count[gloss] += 1
            else:
                X_val.append(img)
                y_val.append(self.gloss_to_numb[gloss])

        # X_train = self.compute_zero_mean(array(X_train))
        # X_val = self.compute_zero_mean(array(X_val))
        # X_test = self.compute_zero_mean(array(X_test))

        # return ((X_train,array(y_train)),(X_val, array(y_val)),(X_test, array(y_test)))
        return ((array(X_train), array(y_train)), (array(X_val), array(y_val)), (array(X_test), array(y_test)))
