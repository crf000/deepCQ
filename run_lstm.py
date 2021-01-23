import numpy as np
from data import Data
from keras.models import Sequential,Model
import os
from model import seg_qua
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.optimizers import SGD,Adam,RMSprop
from keras import backend as K
from Multi_loss_layer import CustomMultiLossLayer2
from keras.callbacks import EarlyStopping
import scipy.io as sio
from argument import my_generator2
import scipy.io as sio
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'



data_path = './old_data.mat'
lab_data = Data(data_path)
net = seg_qua()
# 5折交叉验证
pre_indice = []
for i in range(0,5):

    # 读取数据
    
    train_image, test_image, \
    train_myo, test_myo, \
    train_indice, test_indice = lab_data.split_train_test_base_on_subjects(i)

    train_area, train_dim, train_rwt = np.split(train_indice, [2, 5], axis=-1)
    test_area, test_dim, test_rwt = np.split(test_indice, [2, 5], axis=-1)

    print('subject:', i + 1)
    # !!!!!!!!!!!!!!!!!!!!!!
    print('training:')
    prediction_model = net.BiLSTM_model(i)

    inp = Input(shape=(20, 80, 80, 1), name='inp')
    y1_pred, y2_pred, y3_pred = prediction_model(inp)
    y1_true = Input(shape=(20, 2), name='y1_true')
    y2_true = Input(shape=(20, 3), name='y2_true')
    y3_true = Input(shape=(20, 6), name='y3_true')

    out = CustomMultiLossLayer2(nb_outputs=3)([y1_true, y2_true, y3_true, y1_pred, y2_pred, y3_pred])
    model = Model([inp, y1_true, y2_true, y3_true], out)
    model.compile(optimizer=Adam(lr=0.001), loss=None)

    model.fit_generator(my_generator2(train_image, train_area, train_dim, train_rwt, batch_size=2, seed=i),
                        steps_per_epoch=len(train_image) // 2,
                        validation_data=([test_image, test_area, test_dim, test_rwt], None),
                        epochs=100, verbose=2)
    print([np.exp(K.get_value(log_var[0])) for log_var in model.layers[-1].log_vars])

    # 开始预测
    print('predicting:')
    predict_area, predict_dim, predict_rwt = prediction_model.predict(test_image)
    predict_indice = np.concatenate([predict_area, predict_dim, predict_rwt], axis=-1)
    predict_indice = np.reshape(predict_indice,(-1,11))
    pre_indice.append(predict_indice)


# 评估
pre_indice=np.vstack(pre_indice)
# Errors 's shape = (2900,11)
Errors = np.abs(pre_indice-lab_data.indice)
# 分别转换成真实值
coaf = lab_data.pix_spacing * lab_data.ratio_resize_inverse*80
for j in range(11):
    if j<= 1 :
        Errors[:,j] = Errors[:,j] * coaf * coaf
        pre_indice[:, j] = pre_indice[:, j] * coaf * coaf
    else:
        Errors[:, j] = Errors[:, j] * coaf
        pre_indice[:, j] = pre_indice[:, j] * coaf
MAEs = np.mean(Errors,axis=0)
stds = np.std(Errors,axis=0)
print("MAE:")
print("%.0f+-%.0f" % (MAEs[0], stds[0]))
print("%.0f+-%.0f" % (MAEs[1], stds[1]))
print("%.2f+-%.2f" % (MAEs[2], stds[2]))
print("%.2f+-%.2f" % (MAEs[3], stds[3]))
print("%.2f+-%.2f" % (MAEs[4], stds[4]))
print("%.2f+-%.2f" % (MAEs[5], stds[5]))
print("%.2f+-%.2f" % (MAEs[6], stds[6]))
print("%.2f+-%.2f" % (MAEs[7], stds[7]))
print("%.2f+-%.2f" % (MAEs[8], stds[8]))
print("%.2f+-%.2f" % (MAEs[9], stds[9]))
print("%.2f+-%.2f" % (MAEs[10], stds[10]))
sio.savemat('pre_indice',
            {'indices':pre_indice})

