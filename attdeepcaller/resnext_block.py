import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import activations


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                         kernel_size=kernel_size,
                                                         strides=strides,
                                                         padding=padding,
                                                         data_format=data_format,
                                                         dilation_rate=dilation_rate,
                                                         activation=activations.get(activation),
                                                         use_bias=use_bias,
                                                         kernel_initializer=initializers.get(kernel_initializer),
                                                         bias_initializer=initializers.get(bias_initializer),
                                                         kernel_regularizer=regularizers.get(kernel_regularizer),
                                                         bias_regularizer=regularizers.get(bias_regularizer),
                                                         activity_regularizer=regularizers.get(activity_regularizer),
                                                         kernel_constraint=constraints.get(kernel_constraint),
                                                         bias_constraint=constraints.get(bias_constraint),
                                                         **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out
#  定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, padding='same', use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(5e-4))
############################### 通道注意力机制 ###############################
class ChannelAttention_F(tf.keras.layers.Layer):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention_F, self).__init__()
        self.avg= tf.keras.layers.GlobalAveragePooling2D()
        self.max= tf.keras.layers.GlobalMaxPooling2D()
        self.conv1 = tf.keras.layers.Conv2D(in_planes//ratio, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                   use_bias=True, activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                   use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = tf.keras.layers.Reshape((1, 1, avg.shape[1]))(avg)   # shape (None, 1, 1 feature)
        max = tf.keras.layers.Reshape((1, 1, max.shape[1]))(max)   # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out


############################### 空间注意力机制 ###############################
class SpatialAttention_F(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_F, self).__init__()
        self.conv1 = regularized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)             # 创建一个维度,拼接到一起concat。
        out = self.conv1(out)

        return out
############################### SE注意力机制 ###############################
class SE_BLOCK(tf.keras.layers.Layer):
    def __init__(self, in_planes, ratio=4):
        super(SE_BLOCK, self).__init__()
        self.avg= tf.keras.layers.GlobalAveragePooling2D()
        self.conv1 = tf.keras.layers.Conv2D(in_planes//ratio, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                   use_bias=True, activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                   use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        avg = tf.keras.layers.Reshape((1, 1, avg.shape[1]))(avg)   # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        out = tf.nn.sigmoid(avg_out)


        return out


import math
############################### ECA注意力机制 ###############################
class ECA_BLOCK(tf.keras.layers.Layer):
    def __init__(self, in_planes, gamma = 2, b = 1):
        super(ECA_BLOCK, self).__init__()
        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(in_planes, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg= tf.keras.layers.GlobalAveragePooling2D()
        self.conv1 = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)

    def call(self, inputs):
        in_channel = inputs.shape[-1]


        avg = self.avg(inputs)


        avg = tf.keras.layers.Reshape((in_channel, 1))(avg)   # shape (None, 1, 1 feature)
        avg_out = self.conv1(avg)
        out = tf.nn.sigmoid(avg_out)
        out = tf.keras.layers.Reshape((1, 1, in_channel))(out)


        return out

############################### NEW 注意力机制 ###############################
class NewATT_BLOCK(tf.keras.layers.Layer):
    def __init__(self, in_planes, gamma = 2, b = 1):
        super(NewATT_BLOCK, self).__init__()
        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(in_planes, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg= tf.keras.layers.GlobalAveragePooling2D()
        self.max = tf.keras.layers.GlobalMaxPooling2D()
        self.conv1 = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)

    def call(self, inputs):
        in_channel = inputs.shape[-1]


        avg = self.avg(inputs)
        max = self.max(inputs)


        avg = tf.keras.layers.Reshape((in_channel, 1))(avg)   # shape (None, 1, 1 feature)
        max = tf.keras.layers.Reshape((in_channel, 1))(max)   # shape (None, 1, 1 feature)
        avg_out = self.conv1(avg)
        max_out = self.conv1(max)
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)
        out = tf.keras.layers.Reshape((1, 1, in_channel))(out)


        return out


############################### ResNeXt_BottleNeck###############################

class ResNeXt_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(ResNeXt_BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.group_conv = GroupConv2D(input_channels=filters,
                                      output_channels=filters,
                                      kernel_size=(3, 3),
                                      strides=strides,
                                      padding="same",
                                      groups=groups)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=2 * filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()

        ############################### 注意力机制 ###############################
        # self.ca = ChannelAttention_F(2*filters)
        # self.sa = SpatialAttention_F()
        ############################### SE ###############################
        # self.se = SE_BLOCK(2*filters)

        ############################### ECA ###############################
        # self.eca = ECA_BLOCK(2 * filters)
        ############################### NewATT ###############################
        self.newatt = NewATT_BLOCK(2 * filters)
        self.sa = SpatialAttention_F()
        self.shortcut_conv = tf.keras.layers.Conv2D(filters=2 * filters,
                                                    kernel_size=(1, 1),
                                                    strides=strides,
                                                    padding="same")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        # print(self.ca(x).shape)#(20, 1, 1, 128)
        # print(x.shape)#(20, 30, 11, 128)
        #x = self.ca(x) * x 报错required broadcastable shapes往往是由维度不同造成的
        ############################### 注意力机制 ###############################
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        ############################### SE ###############################
        # x = self.se(x) * x

        ############################### ECA ###############################
        # x = self.eca(x) * x
        ############################### ECA&SAM ###############################
        x = self.newatt(x) * x
        x = self.sa(x) * x

        x = tf.nn.relu(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


def build_ResNeXt_block(filters, strides, groups, repeat_num):
    block = tf.keras.Sequential()
    block.add(ResNeXt_BottleNeck(filters=filters,
                                 strides=strides,
                                 groups=groups))
    for _ in range(1, repeat_num):
        block.add(ResNeXt_BottleNeck(filters=filters,
                                     strides=1,
                                     groups=groups))

    return block