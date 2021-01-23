from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error

# Custom loss layer
class CustomMultiLossLayer1(Layer):
    def __init__(self, nb_outputs=4, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer1, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer1, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        # mask 80 * 80 *
        mask_loss = binary_crossentropy(K.flatten(ys_true[0]), K.flatten(ys_pred[0]))
        mask_precision = K.exp(- 2 * self.log_vars[0][0])
        loss += mask_precision * mask_loss + self.log_vars[0][0]

        #area
        area_precision = K.exp(- 2 * self.log_vars[1][0])
        loss += area_precision * K.mean(K.square(ys_true[1] - ys_pred[1]), -1) / 2. + self.log_vars[1][0]

        # dim
        dim_precision = K.exp(- 2 * self.log_vars[2][0])
        loss += dim_precision * K.mean(K.square(ys_true[2] - ys_pred[2]), -1) / 2. + self.log_vars[2][0]

        # rwt
        rwt_precision = K.exp(- 2 * self.log_vars[3][0])
        loss += rwt_precision * K.mean(K.square(ys_true[3] - ys_pred[3]), -1) / 2. + self.log_vars[3][0]


        #return loss
        return K.mean(loss)

    def call(self, inputs):
        """

        :param inputs: mask area dim rwt(loss:bce mae mse mse)
               0:4是真实值 4:8是预测值
        :return:
        """
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return inputs


class CustomMultiLossLayer2(Layer):
    def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer2, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer2, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        # area
        area_precision = K.exp(- 2 * self.log_vars[0][0])
        loss += area_precision * K.mean(K.square(ys_true[0] - ys_pred[0]), -1) / 2. + self.log_vars[0][0]

        # dim
        dim_precision = K.exp(- 2 * self.log_vars[1][0])
        loss += dim_precision * K.mean(K.square(ys_true[1] - ys_pred[1]), -1) / 2. + self.log_vars[1][0]

        # rwt
        rwt_precision = K.exp(- 2 * self.log_vars[2][0])
        loss += rwt_precision * K.mean(K.square(ys_true[2] - ys_pred[2]), -1) / 2. + self.log_vars[2][0]

        # return loss
        return K.mean(loss, axis=-1)

    def call(self, inputs):
        """

        :param inputs: mask area dim rwt(loss:bce mae mse mse)
               0:3是真实值 3:6是预测值
        :return:
        """
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return inputs