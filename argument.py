from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# 为run_nolstm设置的数据增强器
def my_generator1(train_image, train_myo, train_area, train_dim, train_rwt, batch_size, seed):
    a_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10).flow(train_image, train_area, batch_size, seed=seed)
    b_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10).flow(train_myo, train_dim, batch_size, seed=seed)
    c_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10).flow(train_myo, train_rwt, batch_size, seed=seed)

    while True:
        train_image_batch, train_area_batch = a_generator.next()
        train_myo_batch, train_dim_batch = b_generator.next()
        _, train_rwt_batch = c_generator.next()
        yield [train_image_batch, train_myo_batch, train_area_batch, train_dim_batch, train_rwt_batch], None

# 为run_lstm设置的数据增强器
def my_generator2(train_image, train_area, train_dim, train_rwt, batch_size, seed):
    train_image = np.reshape(train_image, (-1, 80, 80, 1))
    train_area = np.reshape(train_area, (-1, 2))
    train_dim = np.reshape(train_dim, (-1, 3))
    train_rwt = np.reshape(train_rwt, (-1, 6))
    a_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10).flow(train_image, train_area, batch_size * 20, seed=seed)
    b_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10).flow(train_image, train_dim, batch_size * 20, seed=seed)
    c_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10).flow(train_image, train_rwt, batch_size * 20, seed=seed)

    while True:
        train_image_batch, train_area_batch = a_generator.next()
        _, train_dim_batch = b_generator.next()
        _, train_rwt_batch = c_generator.next()

        train_image_batch = np.reshape(train_image_batch, (-1, 20, 80, 80, 1))
        train_area_batch = np.reshape(train_area_batch, (-1, 20, 2))
        train_dim_batch = np.reshape(train_dim_batch, (-1, 20, 3))
        train_rwt_batch = np.reshape(train_rwt_batch, (-1, 20, 6))

        yield [train_image_batch, train_area_batch, train_dim_batch, train_rwt_batch], None