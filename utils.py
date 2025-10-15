import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D, Add, Conv2D, Dense, Flatten, Dropout, LayerNormalization, DepthwiseConv2D


def channel_shuffle(x, groups):
    _, width, height, channels = x.shape
    group_ch = channels // groups
    x = layers.Reshape([width, height, group_ch, groups])(x)
    x = layers.Permute([1, 2, 4, 3])(x)
    x = layers.Reshape([width, height, channels])(x)
    return x



class PoolingLayer(keras.layers.Layer):

    def __init__(self, filters, frac_ratio=1.0 ,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups
        self.frac_ratio = frac_ratio

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        config.update({"frac_ratio": self.frac_ratio})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        if self.frac_ratio == 2.0: self.pool = MaxPool2D(pool_size=(2, 2)) #AveragePooling2D(pool_size=(2, 2))
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, input):
        x = input
        if self.frac_ratio < 2.0 and self.frac_ratio > 0.0: x = tf.nn.fractional_max_pool(value=x, pooling_ratio=[1, self.frac_ratio, self.frac_ratio, 1], pseudo_random=True, overlapping=False)[0]
        elif self.frac_ratio == 2.0: x = self.pool(x) 
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        
        return x

# ---------------------------------------------------------
# Your GroupConv2D (unchanged)
# ---------------------------------------------------------
class GroupConv2D(layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size=(3,3), padding='same', groups=1, **kwargs):
        super().__init__(**kwargs)
        assert input_channels % groups == 0, "in_ch must be divisible by groups"
        assert output_channels % groups == 0, "out_ch must be divisible by groups"
        self.in_ch = input_channels
        self.out_ch = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.convs = []

    def build(self, input_shape):
        for _ in range(self.groups):
            self.convs.append(
                Conv2D(self.out_ch // self.groups, self.kernel_size, padding=self.padding)
            )

    def call(self, x):
        splits = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        outs = [conv(s) for conv, s in zip(self.convs, splits)]
        return tf.concat(outs, axis=-1)

# ---------------------------------------------------------
# Your ResidualBlock (as provided; VALID 3x3s and projection)
# ---------------------------------------------------------
class ResidualBlock3x3(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                 groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                 groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        # NOTE: your projection shrinks spatial dims via 5x5 VALID; assumed to match main path.
        #self.projection = DepthwiseConv2D(kernel_size=[5, 5], padding='valid')
        #self.proj_norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    

class ResidualBlockDepthwise3x3(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = DepthwiseConv2D(kernel_size=[3, 3], padding='same')
        self.norm2 = LayerNormalization()
        self.conv3 = DepthwiseConv2D(kernel_size=[3, 3], padding='same')
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y

class ResidualBlockDepthwise5x5(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(5, 5), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = DepthwiseConv2D(kernel_size=[5, 5], padding='same')
        self.norm2 = LayerNormalization()
        self.conv3 = DepthwiseConv2D(kernel_size=[5, 5], padding='same')
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
class ResidualBlockDepthwise7x7(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(7, 7), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = DepthwiseConv2D(kernel_size=[7, 7], padding='same')
        self.norm2 = LayerNormalization()
        self.conv3 = DepthwiseConv2D(kernel_size=[7, 7], padding='same')
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
class ResidualBlockDepthwise9x9(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(9, 9), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = DepthwiseConv2D(kernel_size=[9, 9], padding='same')
        self.norm2 = LayerNormalization()
        self.conv3 = DepthwiseConv2D(kernel_size=[9, 9], padding='same')
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    

class ResidualBlock5x5(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(5, 5), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // (self.block_reduction*2), kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//(self.block_reduction*2),
                                 output_channels=self.filters//(self.block_reduction*2), 
                                 kernel_size=[5, 5], 
                                 padding='same',
                                 groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//(self.block_reduction*2),
                                 output_channels=self.filters//(self.block_reduction*2), 
                                 kernel_size=[5, 5], 
                                 padding='same',
                                 groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        # NOTE: your projection shrinks spatial dims via 5x5 VALID; assumed to match main path.
        #self.projection = DepthwiseConv2D(kernel_size=[5, 5], padding='valid')
        #self.proj_norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
class ResidualBlock7x7(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // (self.block_reduction * 4), kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//(self.block_reduction * 4),
                                 output_channels=self.filters//(self.block_reduction * 4), 
                                 kernel_size=[7, 7], 
                                 padding='same',
                                 groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//(self.block_reduction * 4),
                                 output_channels=self.filters//(self.block_reduction * 4), 
                                 kernel_size=[7, 7], 
                                 padding='same',
                                 groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        # NOTE: your projection shrinks spatial dims via 5x5 VALID; assumed to match main path.
        #self.projection = DepthwiseConv2D(kernel_size=[5, 5], padding='valid')
        #self.proj_norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
 
class ResidualBlock7x7(keras.layers.Layer):
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // (self.block_reduction * 4), kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//(self.block_reduction * 4),
                                 output_channels=self.filters//(self.block_reduction * 4), 
                                 kernel_size=[7, 7], 
                                 padding='same',
                                 groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//(self.block_reduction * 4),
                                 output_channels=self.filters//(self.block_reduction * 4), 
                                 kernel_size=[7, 7], 
                                 padding='same',
                                 groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        # NOTE: your projection shrinks spatial dims via 5x5 VALID; assumed to match main path.
        #self.projection = DepthwiseConv2D(kernel_size=[5, 5], padding='valid')
        #self.proj_norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x, training=None):
        shortcut = x #self.proj_norm(self.projection(x))
        y = self.relu1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        y = self.relu3(self.norm3(self.conv3(y)))
        y = self.conv4(y)
        y = self.add([shortcut, y])
        y = self.relu(y)
        return y
    
class SpatialSE(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        self.reduction = GroupConv2D(input_channels=input_shape[-1],
                                output_channels=self.ratio, 
                                kernel_size=[5, 5], 
                                padding='same',
                                groups=self.ratio,
                                use_bias=False)
        self.relu = layers.Activation("relu")
        self.attn = Conv2D(1, kernel_size=(7, 7), padding='same', use_bias=False, activation="softmax", kernel_initializer="he_normal")
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.relu(self.reduction(x))
        x = self.attn(x)
        x = self.multiply([shortcut, x])
        return x
    
class ChannelSE(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze_avg = layers.GlobalAveragePooling2D(keepdims=True)
        self.channel_reduction = layers.Dense(
            units=filters // (self.ratio), activation="relu", use_bias=False, kernel_initializer="he_normal"
        )
        self.channel_excite = layers.Dense(units=filters, activation="sigmoid", use_bias=False, kernel_initializer="he_normal") #TRY: softmax
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze_avg(x)
        x = self.channel_reduction(x)
        x = self.channel_excite(x)
        x = self.multiply([shortcut, x])
        return x
    

class TransposeConvBlock(keras.layers.Layer):

    def __init__(self, filters, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"strides": self.strides})
        return config

    def build(self, input_shape):
        kernel_size = 3 if self.strides == 1 else 4
        self.norm = LayerNormalization()
        self.conv = Conv2D(self.filters, kernel_size=(kernel_size, kernel_size), strides=self.strides, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.relu = layers.Activation("swish")

    def call(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x            
    


class ConvBlock(keras.layers.Layer):

    def __init__(self, filters, pool=False, attn=False, kernel=3, stride=(1, 1), dilation=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.attn = attn
        self.pool = pool
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation

    def get_config(self):
        config = super().get_config()
        config.update({"attn": self.attn})
        config.update({"pool": self.pool})
        config.update({"kernel": self.kernel})
        config.update({"stride": self.stride})
        config.update({"filters": self.filters})
        config.update({"dilation": self.dilation})
        return config

    def build(self, input_shape):

        self.norm = LayerNormalization()
        self.conv = Conv2D(self.filters, kernel_size=[self.kernel, self.kernel], strides=self.stride, dilation_rate=self.dilation, padding='same')
        self.relu = layers.Activation("swish")

    def call(self, input, training):
        x = input
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        
        return x            
    
    

class GroupConv2D(keras.layers.Layer):

    def __init__(self, input_channels, output_channels, kernel_size=(3, 3),
                 padding='same', strides=(1, 1), groups=1, use_bias=True, **kwargs):
        super(GroupConv2D, self).__init__(**kwargs)
        if not input_channels % groups == 0:
            raise ValueError("The input channel must be divisible by the no. of groups")
        if not output_channels % groups == 0:
            raise ValueError("The output channel must be divisible by the no. of groups") 
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        self.use_bias = use_bias


    def build(self, input_shape):
        for i in range(self.groups):
            self.conv_list.append(
            Conv2D(filters=self.group_out_num, kernel_size=self.kernel_size, padding=self.padding, strides=self.strides, use_bias=self.use_bias, kernel_initializer="he_normal"))

    def call(self, input):
        feature_map_list = []
        splits = tf.split(input, self.groups, axis=-1)
        for split, conv in zip(splits, self.conv_list):
            feature_map_list.append(conv(split))
        x = keras.layers.concatenate(feature_map_list, axis=-1)
        
        return x
    

class ResidualBlock(keras.layers.Layer):
    
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training):
        shortcut = x
        x = self.relu1(self.conv1(x))
        #x = self.norm2(self.conv2(x))
        #x = channel_shuffle(x, self.groups)
        #x = self.relu3(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        x = self.add([shortcut, x])
        x = self.relu(x)
        return x
    

class FullResidualBlock(keras.layers.Layer):
    
    def __init__(self, filters, block_reduction=1, groups=1, kernel_size=(3, 3), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.block_reduction = block_reduction
        self.groups = groups
        self.kernel_size = kernel_size
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"block_reduction": self.block_reduction})
        config.update({"groups": self.groups})
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // self.block_reduction, kernel_size=[1, 1], padding='same')
        self.relu1 = layers.Activation("swish")
        self.conv2 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                groups=self.groups)
        self.norm2 = LayerNormalization()
        self.conv3 = GroupConv2D(input_channels=self.filters//self.block_reduction,
                                 output_channels=self.filters//self.block_reduction, 
                                 kernel_size=[3, 3], 
                                 padding='same',
                                groups=self.groups) 
        self.norm3 = LayerNormalization()
        self.relu3 = layers.Activation("swish")
        self.conv4 = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.add = Add()
        self.relu = layers.Activation("swish")

    def call(self, x, training):
        shortcut = x
        x = self.relu1(self.conv1(x))
        x = self.norm2(self.conv2(x))
        x = channel_shuffle(x, self.groups)
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        x = self.add([shortcut, x])
        x = self.relu(x)
        return x



class SpatialSE(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        self.reduction = DepthwiseConv2D(kernel_size=[5, 5], padding='same', use_bias=False)
        self.relu = layers.Activation("swish")
        self.attn = Conv2D(1, kernel_size=(7, 7), padding='same', use_bias=False, activation="sigmoid", kernel_initializer="he_normal")
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.relu(self.reduction(x))
        x = self.attn(x)
        x = self.multiply([shortcut, x])
        return x
    

class ChannelSE(layers.Layer):
    """Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze_avg = layers.GlobalAveragePooling2D(keepdims=True)
        self.channel_reduction = layers.Dense(
            units=filters // (self.ratio), activation="relu", use_bias=False, kernel_initializer="he_normal"
        )
        self.channel_excite = layers.Dense(units=filters, activation="sigmoid", use_bias=False, kernel_initializer="he_normal") #TRY: softmax
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze_avg(x)
        x = self.channel_reduction(x)
        x = self.channel_excite(x)
        x = self.multiply([shortcut, x])
        return x


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.query = tf.keras.layers.Conv2D(self.channels // 8, kernel_size=1)
        self.key = tf.keras.layers.Conv2D(self.channels // 8, kernel_size=1)
        self.value = tf.keras.layers.Conv2D(self.channels // 2, kernel_size=1)
        self.output_conv = tf.keras.layers.Conv2D(self.channels, kernel_size=1)

    def call(self, x):
        batch_size, height, width, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        q = self.query(x)  # (batch, height, width, channels // 8)
        k = self.key(x)    # (batch, height, width, channels // 8)
        v = self.value(x)  # (batch, height, width, channels // 2)

        q_flatten = tf.reshape(q, [batch_size, -1, self.channels // 8])  # (batch, height*width, channels // 8)
        k_flatten = tf.reshape(k, [batch_size, -1, self.channels // 8])  # (batch, height*width, channels // 8)
        v_flatten = tf.reshape(v, [batch_size, -1, self.channels // 2])  # (batch, height*width, channels // 2)

        attention_weights = tf.nn.softmax(tf.matmul(q_flatten, k_flatten, transpose_b=True))  # (batch, height*width, height*width)
        attention_out = tf.matmul(attention_weights, v_flatten)  # (batch, height*width, channels // 2)

        attention_out = tf.reshape(attention_out, [batch_size, height, width, self.channels // 2])  # Reshape back
        output = self.output_conv(attention_out) + x  # Residual connection

        return output


class GenResidualBlock(keras.layers.Layer):
    
    def __init__(self, filters, strides=(1, 1),  **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"strides": self.strides})
        return config

    def build(self, input_shape):
        self.conv_1 = Conv2DTranspose(self.filters, kernel_size=3, strides=self.strides, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.shortcut_conv = Conv2DTranspose(self.filters, kernel_size=1, strides=self.strides, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.shortcut_norm = LayerNormalization()

        self.norm_1 = LayerNormalization()
        self.relu_1 = layers.Activation("swish")

        self.conv_2 = Conv2D(self.filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.norm_2 = LayerNormalization()
        self.add = Add()
        self.relu_2 = layers.Activation("swish")

    def call(self, x, training):
        shortcut = self.shortcut_norm(self.shortcut_conv(x))
        x = self.relu_1(self.norm_1(self.conv_1(x)))
        x = self.norm_2(self.conv_2(x))
        x = self.add([shortcut, x])
        x = self.relu_2(x)
        return x


class DiscResidualBlock(keras.layers.Layer):
    
    def __init__(self, filters, kernel_size=(3, 3), drop_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        
    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"kernel_size": self.kernel_size})
        config.update({"drop_rate": self.drop_rate})
        return config

    def build(self, input_shape):
        self.residual = Conv2D(self.filters, kernel_size=2, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.conv1 = layers.SpectralNormalization(Conv2D(filters=self.filters, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer='he_normal'))  #, kernel_constraint=tf.keras.constraints.MaxNorm(2.0)
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.dropout1 = layers.Dropout(self.drop_rate)
        self.conv2 = layers.SpectralNormalization(Conv2D(filters=self.filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal'))  #, kernel_constraint=tf.keras.constraints.MaxNorm(2.0)

        
        self.add = Add()

    def call(self, x, training):
        shortcut = self.residual(x)
        x = self.dropout1(self.relu1(self.conv1(x)))
        x = self.conv2(x)
        x = self.add([shortcut, x])
        return x
    

class TransitionLayer(keras.layers.Layer):

    def __init__(self, filters, groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        return config

    def build(self, input_shape):
        self.conv = GroupConv2D(input_channels=input_shape[-1],
                                 output_channels=self.filters, 
                                 kernel_size=[3, 3], 
                                 strides=(2, 2),
                                 padding='same',
                                groups=self.groups) 
        self.norm = LayerNormalization()
        self.relu = layers.Activation("swish")

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        
        return x
    

class PoolingLayer(keras.layers.Layer):

    def __init__(self, filters, frac_ratio=1.0 ,groups=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.groups = groups
        self.frac_ratio = frac_ratio

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"groups": self.groups})
        config.update({"frac_ratio": self.frac_ratio})
        return config

    def build(self, input_shape):
        self.depth = input_shape[-1]
        if self.frac_ratio == 2.0: self.pool = MaxPool2D(pool_size=(2, 2)) #AveragePooling2D(pool_size=(2, 2))
        self.channel_up_conv = Conv2D(self.filters, kernel_size=[1, 1], padding='same')
        self.channel_up_norm = LayerNormalization()
        self.channel_up_swish = layers.Activation("swish")

    def call(self, input):
        x = input
        if self.frac_ratio < 2.0 and self.frac_ratio > 0.0: x = tf.nn.fractional_max_pool(value=x, pooling_ratio=[1, self.frac_ratio, self.frac_ratio, 1], pseudo_random=True, overlapping=False)[0]
        elif self.frac_ratio == 2.0: x = self.pool(x) 
        x = self.channel_up_conv(x)
        x = self.channel_up_norm(x)
        x = self.channel_up_swish(x)
        
        return x
    
class GlobalAvgPoolLayer(keras.layers.Layer):

    def __init__(self, num_classes, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.drop_rate =  drop_rate

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        config.update({"drop_rate": self.drop_rate})
        return config

    def build(self, input_shape):
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = Dropout(rate=self.drop_rate)
        self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, x):
        x = self.global_avg_pooling(x)
        if self.drop_rate > 0.0: x = self.dropout(x)
        logits = self.dense(x)
        
        return logits
    

class BaseGenerator(keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        return config
    
    def build(self, input_shape):
        self.conv_trans_1 = Conv2DTranspose(256, (4,4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.relu_1 = layers.Activation("swish")
        self.norm_1 = LayerNormalization()
        self.conv_trans_2 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.relu_2 = layers.Activation("swish")
        self.norm_2 = LayerNormalization()
        self.conv_trans_3 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.relu_3 = layers.Activation("swish") 
        self.norm_3 = LayerNormalization()
        self.conv_trans_final = Conv2D(3, (3, 3), strides=(1,1), padding='same', use_bias=False, activation='tanh', kernel_initializer='he_normal')


    def call(self, x):
        x = self.norm_1(self.relu_1(self.conv_trans_1(x))) 
        x = self.norm_2(self.relu_2(self.conv_trans_2(x))) 
        x = self.norm_3(self.relu_3(self.conv_trans_3(x))) 
        x = self.conv_trans_final(x) 
        return x



class BaseDiscriminator(keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        return config
    
    def build(self, input_shape): 
        # TODO try: just use BatchNorm
        self.conv_1 = layers.SpectralNormalization(Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=init))  #, kernel_constraint=tf.keras.constraints.MaxNorm(2.0)
        self.relu_1 = layers.LeakyReLU(alpha=0.2)
        self.dropout_1 = layers.Dropout(0.3)
        self.conv_2 = layers.SpectralNormalization(Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=init))
        self.relu_2 = layers.LeakyReLU(alpha=0.2)
        self.dropout_2 = layers.Dropout(0.3)
        self.conv_3 = layers.SpectralNormalization(Conv2D(filters=256, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=init))
        self.relu_3 = layers.LeakyReLU(alpha=0.2)
        self.dropout_3 = layers.Dropout(0.3)
        self.flatten = Flatten()
        self.dropout = layers.Dropout(0.3)
        self.dense = Dense(1, activation='sigmoid')

        self.concat = layers.Concatenate()
        self.upsample = tf.keras.layers.UpSampling2D(size=(8,8))
    
    def call(self, input, disc_train=True):
        x, label = input[0], input[1]
        label = self.upsample(label)
        x = self.concat([x, label])
        x = self.dropout_1(x)
        x = self.relu_1(self.conv_1(x))
        x = self.dropout_2(x)
        x = self.relu_2(self.conv_2(x))
        x = self.dropout_3(x)
        x = self.relu_3(self.conv_3(x))
        x = self.flatten(x)
        x = self.dropout(x)

        x = self.dense(x)
        return x
    