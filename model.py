from keras.models import Sequential,Model
from keras.layers.core import Dense,Reshape,Lambda
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, Conv2D, UpSampling2D
from keras.layers import Flatten,Dropout,Concatenate
from keras.layers import Activation,TimeDistributed,LSTM,Bidirectional
from keras.optimizers import SGD,Adam,RMSprop
from keras.regularizers import l1,l1_l2
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K


class seg_qua(object):
    def __init__(self):

        self.input = None
        self.input_shape = (80, 80, 1)

        self.seg_model = None
        self.seg_qua_model = None
        #self.pre_seg_qua_model = None

        self.seg_down_l0 = None  # downsample layer 1
        self.seg_down_l1 = None  # downsample layer 2
        self.seg_down_l2 = None  # downsample layer 2
        self.seg_down_l3 = None

        self.bottleneck = None  # most downsampled UNet layer

        self.seg_up_l3 = None
        self.seg_up_l2 = None  # upsample layer 1
        self.seg_up_l1 = None  # upsample layer 1
        self.seg_up_l0 = None  # upsample layer 2

        self.qua_down_l0 = None
        self.qua_down_l1 = None  # upsample layer 1
        self.qua_down_l2 = None  # upsample layer 1
        self.qua_down_l3 = None  # upsample layer 2
        self.qua_down_l4 = None

        self.base_channel = 16


    def get_segqua_model(self):
        input = Input(shape=self.input_shape)
        self.seg_downsample(input, trainable=True)
        self.seg_bottleneck(trainable=True)
        self.seg_upsample(trainable=True)
        mask = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='binary')(self.seg_up_l0)
        mask = Conv2D(1, 1, activation='sigmoid', name='mask')(mask)

        l = self.qua_down(input, trainable=True)
        l = Flatten()(l)
        l = Dense(1000, activation='relu', name='fc1')(l)
        l = Dropout(0.5)(l)

        area = Dense(2, name='area')(l)
        dim = Dense(3, name='dim')(l)
        rwt = Dense(6, name='rwt')(l)

        model = Model(inputs=input, outputs=[mask, area, dim, rwt])

        return model

    def BiLSTM_model(self, i):
        feature_model = self.get_qua_feature_model()
        feature_model.load_weights('base_weight/quaModel_%d.h5' % i, by_name=True)

        input = Input((20, 80, 80, 1))
        x = TimeDistributed(feature_model)(input)
        x = Dropout(0.5)(x)
        x = LSTM(200, return_sequences=True)(x)
        area = TimeDistributed(Dense(2, name='area'))(x)
        dim = TimeDistributed(Dense(3, name='dim'))(x)
        rwt = TimeDistributed(Dense(6, name='rwt'))(x)

        model = Model(input, [area, dim, rwt])
       # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        model.summary()
        return model

    def get_qua_feature_model(self):
        input = Input(shape=self.input_shape)
        self.seg_downsample(input, trainable=False)
        self.seg_bottleneck(trainable=False)
        self.seg_upsample(trainable=False)
        l = self.qua_down(input, trainable=False)
        l = Flatten()(l)
        feature = Dense(1000, activation='relu', name='fc1', trainable=True)(l)
        model = Model(inputs=input, outputs=feature)

        # model.summary()
        return model


    def seg_downsample(self, inp, trainable):

        self.seg_down_l0 = conv_block(l0=inp, output_channels=self.base_channel, name='seg_down_l0', trainable=trainable)

        l = MaxPooling2D(pool_size=(2, 2))(self.seg_down_l0)
        self.seg_down_l1 = conv_block(l0=l, output_channels=2*self.base_channel, name='seg_down_l1', trainable=trainable)

        l = MaxPooling2D(pool_size=(2, 2))(self.seg_down_l1)
        self.seg_down_l2 = conv_block(l0=l, output_channels=4*self.base_channel, name='seg_down_l2', trainable=trainable)

        l = MaxPooling2D(pool_size=(2, 2))(self.seg_down_l2)
        self.seg_down_l3 = conv_block(l0=l, output_channels=8*self.base_channel, name='seg_down_l3', trainable=trainable)

    def seg_bottleneck(self, trainable):

        l = Dropout(0.5)(self.seg_down_l3)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = conv_block(l0=l, output_channels=16*self.base_channel, name='seg_bottleneck', trainable=trainable)
        self.bottleneck = Dropout(0.5)(l)

    def seg_upsample(self, trainable):

        self.seg_up_l3 = up_block(self.bottleneck,self.seg_down_l3, 8*self.base_channel,
                                  name='seg_up_l3', trainable=trainable)
        self.seg_up_l2 = up_block(self.seg_up_l3, self.seg_down_l2, 4*self.base_channel,
                                  name='seg_up_l2', trainable=trainable)
        self.seg_up_l1 = up_block(self.seg_up_l2, self.seg_down_l1, 2*self.base_channel,
                                  name='seg_up_l1', trainable=trainable)
        self.seg_up_l0 = up_block(self.seg_up_l1, self.seg_down_l0, self.base_channel,
                                  name='seg_up_l0', trainable=trainable)


    def qua_down(self, inp, trainable):
        # 80
        l = qua_down_block(inp, self.seg_up_l0, name='qua_down_l0', trainable = trainable)
        # 40
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = qua_down_block(l, self.seg_up_l1, name='qua_down_l1', trainable = trainable)
        # 20
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = qua_down_block(l, self.seg_up_l2, name='qua_down_l2',trainable = trainable)
        # 10
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = qua_down_block(l, self.seg_up_l3, name='qua_down_l3',trainable = trainable)

        # 5
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l = qua_down_block(l, self.bottleneck, name='qua_down_l4',trainable = trainable)


        return l


def conv_block(l0, output_channels, name, trainable):
    l =Conv2D(output_channels, 3, strides=1, padding='same',
              activation='relu', name=name+'_1', trainable=trainable)(l0)
    l = Conv2D(output_channels, 3, strides=1, padding='same',
               activation='relu', name=name+'_2',trainable = trainable)(l)
    return l

def up_block(l0,concat_l, output_channels, name, trainable):
    l = UpSampling2D(size=(2,2))(l0)
    l = Conv2D(output_channels, 3, strides=1, padding='same',
               activation='relu', name=name+'_plus',trainable = trainable)(l)
    l = Concatenate()([l, concat_l])
    l = conv_block(l,output_channels,name, trainable)
    return l


def qua_down_block(l0, concat_l, name, trainable):
    l1 = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', name=name+'_1', trainable = trainable)(concat_l)
    l2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', name=name+'_2', trainable = trainable)(l0)
    l = Concatenate(name=name+'_concat')([l1, l2])
    l = Activation('relu')(l)
    return l

