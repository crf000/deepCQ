import os
import numpy as np
import scipy.io as sio

class Data:
    def __init__(self, data_path):

        load_data = sio.loadmat(data_path)
        self.image = load_data['image']
        self.myo = load_data['myo']
        self.indice = load_data['indice']
        self.pix_spacing = load_data['pix_spacing']
        self.ratio_resize_inverse = load_data['ratio_resize_inverse']
    # 5折交叉验证
    def split_train_test_base_on_imgs(self, i):
        test_image = self.image[i * 580: (i + 1) * 580]
        train_image_1 = self.image[0: i * 580]
        train_image_2 = self.image[(i + 1) * 580:]
        train_image = np.append(train_image_1, train_image_2, axis=0)

        test_myo = self.myo[i * 580:(i+ 1) * 580]
        train_myo_1 = self.myo[0:i*580]
        train_myo_2 = self.myo[(i+1)*580:]
        train_myo = np.append(train_myo_1, train_myo_2, axis=0)

        test_indice = self.indice[i * 580:(i + 1) * 580]
        train_indice_1 = self.indice[0:i * 580]
        train_indice_2 = self.indice[(i + 1) * 580:]
        train_indice = np.append(train_indice_1, train_indice_2, axis=0)

        return train_image, test_image, \
               train_myo, test_myo, \
               train_indice, test_indice

    def split_train_test_base_on_subjects(self, i):
        image = self.image.reshape(145,20, 80, 80, 1)
        myo = self.myo.reshape(145,20, 80, 80, 1)
        indice = self.indice.reshape(145,20,11)


        test_image = image[i * 29: (i + 1) * 29]
        train_image_1 = image[0: i * 29]
        train_image_2 = image[(i + 1) * 29:]
        train_image = np.append(train_image_1, train_image_2, axis=0)

        test_myo = myo[i * 29:(i+ 1) * 29]
        train_myo_1 = myo[0:i*29]
        train_myo_2 = myo[(i+1)*29:]
        train_myo = np.append(train_myo_1, train_myo_2, axis=0)

        test_indice = indice[i * 29:(i + 1) * 29]
        train_indice_1 = indice[0:i * 29]
        train_indice_2 = indice[(i + 1) * 29:]
        train_indice = np.append(train_indice_1, train_indice_2, axis=0)

        return train_image, test_image,\
               train_myo, test_myo, \
               train_indice, test_indice
