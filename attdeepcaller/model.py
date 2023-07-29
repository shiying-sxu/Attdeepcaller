import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    import tensorflow as tf
import logging
import numpy as np
logging.basicConfig(format='%(message)s', level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

from attdeepcaller.task.main import GT21, GENOTYPE, VARIANT_LENGTH_1, VARIANT_LENGTH_2
import shared.param_f as param

from attdeepcaller.resnext_block import build_ResNeXt_block
import numpy as np
np.random.seed(0)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



params = dict(
            float_type=tf.float32,
            task_loss_weights=[
                1,                       # gt21
                1,                       # genotype
                1,                       # variant/indel length 0
                1,                       # variant/indel length 1
                1                        # l2 loss
            ],
            output_shape=GT21.output_label_count + \
                         GENOTYPE.output_label_count + \
                         VARIANT_LENGTH_1.output_label_count + \
                         VARIANT_LENGTH_2.output_label_count,
            output_gt21_shape=GT21.output_label_count,
            output_genotype_shape=GENOTYPE.output_label_count,
            output_indel_length_shape_1=VARIANT_LENGTH_1.output_label_count,
            output_indel_length_shape_2=VARIANT_LENGTH_2.output_label_count,
            output_gt21_entropy_weights=[1] * GT21.output_label_count,
            output_genotype_entropy_weights=[1] * GENOTYPE.output_label_count,
            output_indel_length_entropy_weights_1=[1] * VARIANT_LENGTH_1.output_label_count,
            output_indel_length_entropy_weights_2=[1] * VARIANT_LENGTH_2.output_label_count,
            L3_dropout_rate=0.2,
            L4_num_units=256,
            L4_pileup_num_units=128,
            L4_dropout_rate=0.5,
            L5_1_num_units=128,
            L5_1_dropout_rate=0.2,
            L5_2_num_units=128,
            L5_2_dropout_rate=0.2,
            L5_3_num_units=128,
            L5_3_dropout_rate=0.2,
            L5_4_num_units=128,
            L5_4_dropout_rate=0.2,
            LSTM1_num_units=128,
            LSTM2_num_units=64,#160
            LSTM1_dropout_rate=0,
            LSTM2_dropout_rate=0.5,
            l2_regularization_lambda=param.l2RegularizationLambda,
        )

add_l2_regulation = True
L2_regularizers = tf.keras.regularizers.l2(params['l2_regularization_lambda']) if add_l2_regulation else None


class BasicConv2D_P(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, SeparableConv=False):
        super(BasicConv2D_P, self).__init__()
        conv = tf.keras.layers.SeparableConv2D if SeparableConv else tf.keras.layers.Conv2D
        self.conv = conv(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           kernel_regularizer=L2_regularizers)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        # inputs = tf.expand_dims(inputs, axis=3)#####将3维的tensor输入扩展维4维的输入
        # inputs = tf.concat([tf.expand_dims(t, axis=3) for t in tf.split(inputs, num_or_size_splits=8, axis=0)], axis=3)
        output = self.conv(inputs)
        output = self.bn(output)
        output = self.relu(output)

        return output
class PyramidPolling_P(tf.keras.layers.Layer):
    def __init__(self, spatial_pool_size=(5, 4, 3, 2, 1)):
        super(PyramidPolling_P, self).__init__()

        self.spatial_pool_size = spatial_pool_size
        self.pool_len = len(self.spatial_pool_size)
        self.window_h = np.empty(self.pool_len, dtype=int)
        self.stride_h = np.empty(self.pool_len, dtype=int)
        self.window_w = np.empty(self.pool_len, dtype=int)
        self.stride_w = np.empty(self.pool_len, dtype=int)

        self.flatten = tf.keras.layers.Flatten()
       ###############################################################



    def build(self, input_shape):
        height = int(input_shape[1])
        width = int(input_shape[2])

        for i in range(self.pool_len):
            self.window_h[i] = self.stride_h[i] = int(np.ceil(height / self.spatial_pool_size[i]))
            self.window_w[i] = self.stride_w[i] = int(np.ceil(width / self.spatial_pool_size[i]))



    def call(self, x):
        for i in range(self.pool_len):
            max_pool = tf.nn.max_pool(x,ksize=[1, self.window_h[i], self.window_w[i], 1],strides=[1, self.stride_h[i], self.stride_w[i], 1],padding='SAME')


            if i == 0:
                pp = self.flatten(max_pool)

            else:
                pp = tf.concat([pp, self.flatten(max_pool)], axis=-1)

        return pp
class BasicConv2D_P2(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, SeparableConv=False):
        super(BasicConv2D_P2, self).__init__()
        conv = tf.keras.layers.SeparableConv2D if SeparableConv else tf.keras.layers.Conv2D
        self.conv = conv(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           kernel_regularizer=L2_regularizers)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        output = self.conv(inputs)
        output = self.bn(output)
        output = self.relu(output)

        return output

class attdeepcaller_P(tf.keras.Model):
    # CBAM-RENET model for attdeepcaller pileup input


    def __init__(self,add_indel_length=False, predict=False):
        super(attdeepcaller_P, self).__init__()

        # output
        self.output_gt21_shape = params['output_gt21_shape']
        self.output_genotype_shape = params['output_genotype_shape']
        self.output_indel_length_shape_1 = params['output_indel_length_shape_1']
        self.output_indel_length_shape_2 = params['output_indel_length_shape_2']

        self.L3_dropout_rate = params['L3_dropout_rate']
        self.L4_num_units = params['L4_num_units']
        self.L4_pileup_num_units = params['L4_pileup_num_units']
        self.L4_dropout_rate = params['L4_dropout_rate']
        self.L5_1_num_units = params['L5_1_num_units']
        self.L5_1_dropout_rate = params['L5_1_dropout_rate']
        self.L5_2_num_units = params['L5_2_num_units']
        self.L5_2_dropout_rate = params['L5_2_dropout_rate']
        self.L5_3_num_units = params['L5_3_num_units']
        self.L5_3_dropout_rate = params['L5_3_dropout_rate']
        self.L5_4_num_units = params['L5_4_num_units']
        self.L5_4_dropout_rate = params['L5_4_dropout_rate']
        self.LSTM1_num_units = params['LSTM1_num_units']
        self.LSTM2_num_units = params['LSTM2_num_units']
        self.LSTM1_dropout_rate = params['LSTM1_dropout_rate']
        self.LSTM2_dropout_rate = params['LSTM2_dropout_rate']

        self.output_label_split = [
            self.output_gt21_shape,
            self.output_genotype_shape,
            self.output_indel_length_shape_1,
            self.output_indel_length_shape_2
        ]

        self.add_indel_length = add_indel_length
        self.predict = predict

        self.LSTM1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=self.LSTM1_num_units,
            return_sequences=True,
            kernel_regularizer=L2_regularizers
        ))

        self.LSTM2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=self.LSTM2_num_units,
            return_sequences=True,
            kernel_regularizer=L2_regularizers
        ))



        self.conv1 = BasicConv2D_P2(filters=64,
                                 kernel_size=(3, 3),
                                 strides=3,
                                 padding="same", )
        # resnext  begining**************
        self.res_block1 = build_ResNeXt_block(filters=64,
                                              strides=1,
                                              groups=8,
                                              repeat_num=2)
        self.conv3 = BasicConv2D_P2(filters=128,
                                 kernel_size=(3, 3),
                                 strides=4,
                                 padding="same")
        self.res_block2 = build_ResNeXt_block(filters=128,
                                              strides=2,
                                              groups=8,
                                              repeat_num=2)
        self.conv5 = BasicConv2D_P2(filters=256,
                                 kernel_size=(3, 3),
                                 strides=6,
                                 padding="same")
        self.res_block3 = build_ResNeXt_block(filters=256,
                                              strides=2,
                                              groups=8,
                                              repeat_num=2)
        self.conv7 = BasicConv2D_P2(filters=512,
                                 kernel_size=(3, 3),
                                 strides=3,
                                 padding="same")
        self.res_block4 = build_ResNeXt_block(filters=512,
                                              strides=2,
                                              groups=8,
                                              repeat_num=2)
        # resnext   ending**************


        self.pyramidpolling = PyramidPolling_P()





        self.L3_dropout = tf.keras.layers.Dropout(rate=self.L3_dropout_rate)

        self.L3_dropout_flatten = tf.keras.layers.Flatten()

        self.L4 = tf.keras.layers.Dense(units=self.L4_pileup_num_units, activation='selu',kernel_regularizer=L2_regularizers)

        self.L4_dropout = tf.keras.layers.Dropout(rate=self.LSTM2_dropout_rate, seed=param.OPERATION_SEED)

        self.L5_1 = tf.keras.layers.Dense(units=self.L5_1_num_units, activation='selu', kernel_regularizer=L2_regularizers)

        self.L5_1_dropout = tf.keras.layers.Dropout(rate=self.L5_1_dropout_rate, seed=param.OPERATION_SEED)

        self.L5_2 = tf.keras.layers.Dense(units=self.L5_2_num_units, activation='selu', kernel_regularizer=L2_regularizers)

        self.L5_2_dropout = tf.keras.layers.Dropout(rate=self.L5_2_dropout_rate, seed=param.OPERATION_SEED)

        self.Y_gt21_logits = tf.keras.layers.Dense(units=self.output_gt21_shape, activation='selu', kernel_regularizer=L2_regularizers)

        self.Y_genotype_logits = tf.keras.layers.Dense(units=self.output_genotype_shape, activation='selu', kernel_regularizer=L2_regularizers)

        if self.add_indel_length:

            self.L5_3 = tf.keras.layers.Dense(units=self.L5_3_num_units, activation='selu', kernel_regularizer=L2_regularizers)

            self.L5_3_dropout = tf.keras.layers.Dropout(rate=self.L5_3_dropout_rate, seed=param.OPERATION_SEED)

            self.L5_4 = tf.keras.layers.Dense(units=self.L5_4_num_units, activation='selu', kernel_regularizer=L2_regularizers)

            self.L5_4_dropout = tf.keras.layers.Dropout(rate=self.L5_4_dropout_rate, seed=param.OPERATION_SEED)

            self.Y_indel_length_logits_1 = tf.keras.layers.Dense(units=self.output_indel_length_shape_1, activation='selu', kernel_regularizer=L2_regularizers)

            self.Y_indel_length_logits_2 = tf.keras.layers.Dense(units=self.output_indel_length_shape_2, activation='selu', kernel_regularizer=L2_regularizers)

        self.softmax = tf.keras.layers.Softmax()




    def call(self, x,):

        # print(x)#shape=(20, 33, 18), dtype=int32)
        # x = tf.cast(x, tf.float32) / param.NORMALIZE_NUM#函数的作用是执行 tensorflow 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
        x = tf.cast(x, tf.float32)
      # ######绘制热图
      #   sess = tf.compat.v1.Session()
      #   with sess.as_default():
      #       # print(type(x.numpy()))#<class 'numpy.ndarray'>
      #       plot1=x.numpy()
      #   fig1=np.reshape(plot1,(360,33))###三维数组转变成二维数组
      #   #对reshape()里面的数组形状第一个为-1，第二个为第二维元素的数目。那么就会对原来的所有元素进行一个平均分配得到一个二维数组。
      #   f, ax = plt.subplots(figsize=(9, 6))
      #   ax = sns.heatmap(fig1)
      #   plt.show()
      #   f.savefig('sns_heatmap_fmt1.jpg')
      #   mask = np.zeros_like(fig1)
      #   mask[np.triu_indices_from(mask)] = True
      #   with sns.axes_style("white"):
      #       ax = sns.heatmap(fig1, mask=mask, vmax=.3, square=True)
      #   plt.show()#仅绘制矩阵的一部分
      #   f.savefig('sns_heatmap_fmt2.jpg')
      #   ######绘制热图end

        # print(x)#shape=(20, 33, 18), dtype=float32)


        x = self.LSTM1(x)  # (batch_size, inp_seq_len, d_model)
        x = self.LSTM2(x)
        # print(x.shape)#(20, 33, 320)

        x = tf.concat([tf.expand_dims(t, axis=3) for t in tf.split(x, num_or_size_splits=1, axis=0)], axis=3)
        # print(x.shape)#(20, 33, 320, 1)
        x = self.conv1(x)
        # print(x.shape)#(20, 11, 107, 64)
        # print(x)#shape=(20, 11, 107, 64), dtype=float32)
        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.conv5(x)
        x = self.res_block3(x)
        x = self.conv7(x)
        x = self.res_block4(x)


        x = self.pyramidpolling(x)


        x = self.L3_dropout(x)

        x = self.L3_dropout_flatten(x)

        x = self.L4(x)

        x = self.L4_dropout(x)

        l5_1_dropout = self.L5_1_dropout(self.L5_1(x))

        l5_2_dropout = self.L5_2_dropout(self.L5_2(x))

        y_gt21_logits = self.softmax(self.Y_gt21_logits(l5_1_dropout))

        y_genotype_logits = self.softmax(self.Y_genotype_logits(l5_2_dropout))

        if self.add_indel_length:
            l5_3_dropout = self.L5_3_dropout(self.L5_3(x))

            l5_4_dropout = self.L5_4_dropout(self.L5_4(x))

            y_indel_length_logits_1 = self.softmax(self.Y_indel_length_logits_1(l5_3_dropout))

            y_indel_length_logits_2 = self.softmax(self.Y_indel_length_logits_2(l5_4_dropout))

            if self.predict:
                return tf.concat([y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2], axis=1)

            return [y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2]

        if self.predict:
            return tf.concat([y_gt21_logits, y_genotype_logits],axis=1)

        return [y_gt21_logits, y_genotype_logits]



class BasicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, SeparableConv=False):
        super(BasicConv2D, self).__init__()
        conv = tf.keras.layers.SeparableConv2D if SeparableConv else tf.keras.layers.Conv2D
        self.conv = conv(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           kernel_regularizer=L2_regularizers)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        output = self.conv(inputs)
        output = self.bn(output)
        output = self.relu(output)

        return output




class PyramidPolling(tf.keras.layers.Layer):
    def __init__(self, spatial_pool_size=(4, 3, 2, 1)):
        super(PyramidPolling, self).__init__()

        self.spatial_pool_size = spatial_pool_size
        self.pool_len = len(self.spatial_pool_size)
        self.window_h = np.empty(self.pool_len, dtype=int)
        self.stride_h = np.empty(self.pool_len, dtype=int)
        self.window_w = np.empty(self.pool_len, dtype=int)
        self.stride_w = np.empty(self.pool_len, dtype=int)



        self.flatten = tf.keras.layers.Flatten()

    def build(self, input_shape):
        height = int(input_shape[1])
        width = int(input_shape[2])


        for i in range(self.pool_len):
            self.window_h[i] = self.stride_h[i] = int(np.ceil(height / self.spatial_pool_size[i]))
            self.window_w[i] = self.stride_w[i] = int(np.ceil(width / self.spatial_pool_size[i]))

    def call(self, x):

        for i in range(self.pool_len):
            max_pool = tf.nn.max_pool(x,ksize=[1, self.window_h[i], self.window_w[i], 1],strides=[1, self.stride_h[i], self.stride_w[i], 1],padding='SAME')
            if i == 0:
                pp = self.flatten(max_pool)

            else:
                pp = tf.concat([pp, self.flatten(max_pool)], axis=-1)

        return pp


class attdeepcaller_F(tf.keras.Model):
    # Residual CNN model for attdeepcaller full alignment input
    def __init__(self, add_indel_length=False, predict=False):
        super(attdeepcaller_F, self).__init__()
        self.output_gt21_shape = params['output_gt21_shape']
        self.output_genotype_shape = params['output_genotype_shape']
        self.output_indel_length_shape_1 = params['output_indel_length_shape_1']
        self.output_indel_length_shape_2 = params['output_indel_length_shape_2']

        self.L3_dropout_rate = params['L3_dropout_rate']
        self.L4_num_units = params['L4_num_units']
        self.L4_dropout_rate = params['L4_dropout_rate']
        self.L5_1_num_units = params['L5_1_num_units']
        self.L5_1_dropout_rate = params['L5_1_dropout_rate']
        self.L5_2_num_units = params['L5_2_num_units']
        self.L5_2_dropout_rate = params['L5_2_dropout_rate']
        self.L5_3_num_units = params['L5_3_num_units']
        self.L5_3_dropout_rate = params['L5_3_dropout_rate']
        self.L5_4_num_units = params['L5_4_num_units']
        self.L5_4_dropout_rate = params['L5_4_dropout_rate']

        self.LSTM1_num_units = params['LSTM1_num_units']
        self.LSTM2_num_units = params['LSTM2_num_units']
        self.LSTM1_dropout_rate = params['LSTM1_dropout_rate']
        self.LSTM2_dropout_rate = params['LSTM2_dropout_rate']

        self.output_label_split = [
            self.output_gt21_shape,
            self.output_genotype_shape,
            self.output_indel_length_shape_1,
            self.output_indel_length_shape_2
        ]

        self.add_indel_length = add_indel_length
        self.predict = predict

        self.LSTM1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=self.LSTM1_num_units,
            return_sequences=True,
            kernel_regularizer=L2_regularizers
        ))

        self.LSTM2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=self.LSTM2_num_units,
            return_sequences=True,
            kernel_regularizer=L2_regularizers
        ))

        self.conv1 = BasicConv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=3,
                                 padding="same",)
        # resnext  begining**************
        self.res_block1 = build_ResNeXt_block(filters=64,
                                          strides=1,
                                          groups=8,
                                          repeat_num=2)
        self.conv3 = BasicConv2D(filters=128,
                                 kernel_size=(3, 3),
                                 strides=4,
                                 padding="same")
        self.res_block2 = build_ResNeXt_block(filters=128,
                                          strides=2,
                                          groups=8,
                                          repeat_num=2)
        self.conv5 = BasicConv2D(filters=256,
                                 kernel_size=(3, 3),
                                 strides=6,
                                 padding="same")
        self.res_block3 = build_ResNeXt_block(filters=256,
                                          strides=2,
                                          groups=8,
                                          repeat_num=2)
        self.conv7 = BasicConv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=3,
                                 padding="same")
        self.res_block4 = build_ResNeXt_block(filters=512,
                                          strides=2,
                                          groups=8,
                                          repeat_num=2)
        # resnext   ending**************


        self.pyramidpolling = PyramidPolling()
        # self.pyramidpolling = ASPP(512)

        self.L3_dropout = tf.keras.layers.Dropout(rate=self.L3_dropout_rate)

        self.flatten = tf.keras.layers.Flatten()

        self.L4 = tf.keras.layers.Dense(units=self.L4_num_units, activation='selu',kernel_regularizer=L2_regularizers)

        self.L4_dropout = tf.keras.layers.Dropout(rate=self.L4_dropout_rate, seed=param.OPERATION_SEED)

        self.L5_1 = tf.keras.layers.Dense(units=self.L5_1_num_units, activation='selu', kernel_regularizer=L2_regularizers)

        self.L5_1_dropout = tf.keras.layers.Dropout(rate=self.L5_1_dropout_rate, seed=param.OPERATION_SEED)

        self.L5_2 = tf.keras.layers.Dense(units=self.L5_1_num_units, activation='selu', kernel_regularizer=L2_regularizers)

        self.L5_2_dropout = tf.keras.layers.Dropout(rate=self.L5_2_dropout_rate, seed=param.OPERATION_SEED)

        self.Y_gt21_logits = tf.keras.layers.Dense(units=self.output_gt21_shape, activation='selu', kernel_regularizer=L2_regularizers)

        self.Y_genotype_logits = tf.keras.layers.Dense(units=self.output_genotype_shape, activation='selu', kernel_regularizer=L2_regularizers)

        if self.add_indel_length:
            self.L5_3 = tf.keras.layers.Dense(units=self.L5_3_num_units, activation='selu', kernel_regularizer=L2_regularizers)

            self.L5_3_dropout = tf.keras.layers.Dropout(rate=self.L5_3_dropout_rate, seed=param.OPERATION_SEED)

            self.L5_4 = tf.keras.layers.Dense(units=self.L5_4_num_units, activation='selu', kernel_regularizer=L2_regularizers)

            self.L5_4_dropout = tf.keras.layers.Dropout(rate=self.L5_4_dropout_rate, seed=param.OPERATION_SEED)

            self.Y_indel_length_logits_1 = tf.keras.layers.Dense(units=self.output_indel_length_shape_1, activation='selu',kernel_regularizer=L2_regularizers)

            self.Y_indel_length_logits_2 = tf.keras.layers.Dense(units=self.output_indel_length_shape_2, activation='selu',kernel_regularizer=L2_regularizers)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):

        # print(inputs)#shape=(20, 89, 33, 8), dtype=int8)
        # print(inputs.shape)#(20, 89, 33, 8)

        x = tf.cast(inputs, tf.float32) / param.NORMALIZE_NUM#tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32

        x = tf.concat([tf.squeeze(t, axis=3) for t in tf.split(x, num_or_size_splits=8, axis=3)],axis=0)
        # print(x.shape)#(160, 89, 33)

        # ######绘制热图
        # sess = tf.compat.v1.Session()
        # with sess.as_default():
        #     # print(type(x.numpy()))#<class 'numpy.ndarray'>
        #     plot1 = inputs.numpy()
        # fig1 = np.reshape(plot1, (660, 712))  ###4维数组转变成二维数组
        # # 对reshape()里面的数组形状第一个为-1，第二个为第二维元素的数目。那么就会对原来的所有元素进行一个平均分配得到一个二维数组。
        # f, ax = plt.subplots(figsize=(9, 6))
        # ax = sns.heatmap(fig1)
        # plt.show()
        # f.savefig('sns_heatmap_fmt-full.jpg')
        # mask = np.zeros_like(fig1)
        # mask[np.triu_indices_from(mask)] = True
        # with sns.axes_style("white"):
        #     ax = sns.heatmap(fig1, mask=mask, vmax=.3, square=True)
        # plt.show()  # 仅绘制矩阵的一部分
        # f.savefig('sns_heatmap_fmt-full2.jpg')
        # ######绘制热图end


        x = self.LSTM1(x)  # (batch_size, inp_seq_len, d_model)

        x = self.LSTM2(x)
        x = tf.concat([tf.expand_dims(t, axis=3) for t in tf.split(x, num_or_size_splits=8, axis=0)], axis=3)


        x = self.conv1(x)

        # print(x)#shape=(20, 30, 11, 64), dtype=float32)在未添加BILSTM之前的输出

        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.conv5(x)
        x = self.res_block3(x)
        x = self.conv7(x)
        x = self.res_block4(x)
        # print(x.shape)  #(None, None, 1, 1024)

        x = self.pyramidpolling(x)
        # print(x.shape)  #(None, None)
        x = self.flatten(self.L3_dropout(x))

        x = self.L4(x)
        x = self.L4_dropout(x)

        l5_1_dropout = self.L5_1_dropout(self.L5_1(x))

        l5_2_dropout = self.L5_2_dropout(self.L5_2(x))

        y_gt21_logits = self.softmax(self.Y_gt21_logits(l5_1_dropout))

        y_genotype_logits = self.softmax(self.Y_genotype_logits(l5_2_dropout))

        if self.add_indel_length:

            l5_3_dropout = self.L5_3_dropout(self.L5_3(x))

            l5_4_dropout = self.L5_4_dropout(self.L5_4(x))

            y_indel_length_logits_1 = self.softmax(self.Y_indel_length_logits_1(l5_3_dropout))

            y_indel_length_logits_2 = self.softmax(self.Y_indel_length_logits_2(l5_4_dropout))

            if self.predict:

                return tf.concat([y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2], axis=1)

            return [y_gt21_logits, y_genotype_logits, y_indel_length_logits_1, y_indel_length_logits_2]

        if self.predict:

            return tf.concat([y_gt21_logits, y_genotype_logits],axis=1)

        return [y_gt21_logits, y_genotype_logits]




