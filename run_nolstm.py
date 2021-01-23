import numpy as np
from keras.models import Sequential,Model
from data import Data
import os
from model import seg_qua
import matplotlib.pyplot as plt
from Multi_loss_layer import CustomMultiLossLayer1
from keras.layers import Input
from keras.optimizers import SGD,Adam,RMSprop
from keras import backend as K
from argument import my_generator1
import scipy.io as sio

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

data_path = './old_data.mat'
lab_data = Data(data_path)
net = seg_qua()
# 5折交叉验证
pre_indice = []
pre_myo = []
for i in range(0, 5):

    # 读取数据

    train_image, test_image, \
    train_myo, test_myo, \
    train_indice, test_indice = lab_data.split_train_test_base_on_imgs(i)

    train_area, train_dim, train_rwt = np.split(train_indice, [2, 5], axis=-1)
    test_area, test_dim, test_rwt = np.split(test_indice, [2, 5], axis=-1)

    #!!!!!!!!!!!!!!!!!!!!!!
    prediction_model = net.get_segqua_model()
    prediction_model.load_weights('./unet_weight/segModel_%d.h5' % i, by_name=True)

    inp = Input(shape=(80, 80, 1), name='inp')
    y1_pred, y2_pred, y3_pred, y4_pred = prediction_model(inp)
    y1_true = Input(shape=(80, 80, 1), name='y1_true')
    y2_true = Input(shape=(2,), name='y2_true')
    y3_true = Input(shape=(3,), name='y3_true')
    y4_true = Input(shape=(6,), name='y4_true')

    out = CustomMultiLossLayer1(nb_outputs=4)([y1_true, y2_true,y3_true, y4_true,
                                              y1_pred, y2_pred, y3_pred, y4_pred])
    model = Model([inp, y1_true, y2_true,y3_true, y4_true], out)
    model.compile(optimizer=Adam(), loss=None)

    print('subject:', i + 1)
    model.fit_generator(my_generator1(train_image, train_myo, train_area, train_dim, train_rwt, batch_size=32, seed=i),
                        steps_per_epoch=len(train_image) // 32,
                        validation_data=([test_image, test_myo, test_area, test_dim, test_rwt], None),
                        epochs=160, verbose=2)
    print([np.exp(K.get_value(log_var[0])) for log_var in model.layers[-1].log_vars])
    prediction_model.save_weights('./base_weight/quaModel_%d.h5' % i)

    # 开始预测
    print('predicting:')
    predict_myo, predict_area, predict_dim, predict_rwt = prediction_model.predict(test_image)
    predict_indice = np.concatenate([predict_area, predict_dim, predict_rwt], axis=-1)
    pre_indice.append(predict_indice)
    pre_myo.append(predict_myo)

# 评估
pre_indice = np.vstack(pre_indice)
pre_myo = np.vstack(pre_myo)


# Errors 's shape = (2900,11)
Errors = np.abs(pre_indice - lab_data.indice)
# 分别转换成真实值
coaf = lab_data.pix_spacing * lab_data.ratio_resize_inverse * 80
for j in range(11):
    if j <= 1:
        Errors[:, j] = Errors[:, j] * coaf * coaf
    else:
        Errors[:, j] = Errors[:, j] * coaf
MAEs = np.mean(Errors, axis=0)
stds = np.std(Errors, axis=0)
print("MAE:")
print("%.0f+-%.0f" % (MAEs[0], stds[0]))
print("%.0f+-%.0f" % (MAEs[1], stds[1]))
ares_together = np.concatenate([Errors[:, 0], Errors[:, 1]])
ares_avg = np.mean(Errors[:, 0:2], axis = 1)
print("%.0f+-%.0f" % (np.mean(ares_avg), np.std(ares_avg)))

print("%.2f+-%.2f" % (MAEs[2], stds[2]))
print("%.2f+-%.2f" % (MAEs[3], stds[3]))
print("%.2f+-%.2f" % (MAEs[4], stds[4]))
dims_avg = np.mean(Errors[:, 2:5], axis = 1)
print("%.2f+-%.2f" % (np.mean(dims_avg), np.std(dims_avg)))

print("%.2f+-%.2f" % (MAEs[5], stds[5]))
print("%.2f+-%.2f" % (MAEs[6], stds[6]))
print("%.2f+-%.2f" % (MAEs[7], stds[7]))
print("%.2f+-%.2f" % (MAEs[8], stds[8]))
print("%.2f+-%.2f" % (MAEs[9], stds[9]))
print("%.2f+-%.2f" % (MAEs[10], stds[10]))
rwts_avg = np.mean(Errors[:, 5:11], axis = 1)
print("%.2f+-%.2f" % (np.mean(rwts_avg), np.std(rwts_avg)))
sio.savemat('pre_indice2',
            {'indices':pre_indice})

###############
#阈值处理

pre_myo = pre_myo[:,:,:,0]
pre_myo = np.round(pre_myo)


dices = np.zeros((2900,)).astype('float32')
for i in range(2900):
    pre = pre_myo[i, :, :]
    true = lab_data.myo[i, :, :, 0]
    intersection = np.sum(pre * true)
    union = np.sum(pre) + np.sum(true)
    dice = (2. * intersection) / (union)
    dices[i] = dice

avg_dice = np.mean(dices)
std_dice = np.std(dices)
print("dice: %.4f(%.4f)" % (avg_dice,std_dice))

sio.savemat('pre_seg',
            {'myo':pre_myo,
             'dices':dices})

