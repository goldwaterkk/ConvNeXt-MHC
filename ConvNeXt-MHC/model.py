import keras.layers
import numpy as np
import tensorflow as tf
from keras import layers, initializers, Model
from tensorflow import sigmoid

KERNEL_INITIALIZER = {
    "class_name": "TruncatedNormal",
    "config": {
        "stddev": 0.2
    }
}

BIAS_INITIALIZER = "Zeros"


class Attention_Layer(layers.Layer):
    # input = 20*9*21  ==  20 * [9*21]
    def __init__(self, name: str = None, init_value={}):  # dim[0] = 32
        super().__init__(name=name)
        self.layer_scale_init_value = 1e-6
        # aa_list = A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V
        self.conv_20_block = [
            layers.Conv1D(filters=1, kernel_size=9, activation=sigmoid,
                          kernel_initializer=keras.initializers.Constant(value=init_value[i])
                          )
            for i in
            ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def build(self, input_shape):
        if self.layer_scale_init_value > 0:
            self.gamma = self.add_weight(shape=[input_shape[-1]],
                                         initializer=initializers.Constant(self.layer_scale_init_value),
                                         trainable=True,
                                         dtype=tf.float32,
                                         name="gamma")
        else:
            self.gamma = None

    def call(self, x, training=False):
        print(x.shape)
        ans_att = []
        for i, conv in enumerate(self.conv_20_block):
            att = conv(x[:, i, :, :])
            ans_att.append(att)
            # [20,batch_size,1,1]
        return ans_att


class Attention(layers.Layer):
    # input = 20*9*21  ==  20 * [9*21]
    def __init__(self):  # dim[0] = 32
        super().__init__()
        self.Att_Block = Attention_Layer(
            init_value={'P': 0, 'Y': 1.0, 'F': 0.032992930086410056, 'A': 0.2875098193244305, 'M': 0.05420267085624509,
                        'Q': 0.24823252160251374, 'E': 0.23644933228593873, 'R': 0.2199528672427337,
                        'N': 0.23095051060487037, 'H': 0.07855459544383346, 'T': 0.42498036135113904,
                        'D': 0.1681068342498036, 'L': 0.2474469756480754, 'I': 0.11626080125687353,
                        'K': 0.16496465043205027, 'W': 0.2513747054202671, 'V': 0.14846818538884524,
                        'G': 0.02199528672427337, 'C': 0.0, 'S': 0.14611154752553024})

    def call(self, x, training=False):
        attention = self.Att_Block(x)
        attention = tf.transpose(attention, perm=(1, 0, 2, 3))
        return attention


# [20, 9, 21]
# [2*2*5, 3*3*3 , 21]

class Block(layers.Layer):
    """
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    # 5 * 3
    def __init__(self, dim, drop_rate=0.2, layer_scale_init_value=1e-6, name: str = None):
        super().__init__(name=name)
        self.layer_scale_init_value = layer_scale_init_value
        self.dwconv = layers.DepthwiseConv2D((7, 3),
                                             padding="same",
                                             depthwise_initializer=KERNEL_INITIALIZER,
                                             bias_initializer=BIAS_INITIALIZER,
                                             name="dwconv")
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.pwconv1 = layers.Dense(4 * dim,
                                    kernel_initializer=KERNEL_INITIALIZER,
                                    bias_initializer=BIAS_INITIALIZER,
                                    name="pwconv1")
        self.act = layers.Activation("gelu")
        self.pwconv2 = layers.Dense(dim,
                                    kernel_initializer=KERNEL_INITIALIZER,
                                    bias_initializer=BIAS_INITIALIZER,
                                    name="pwconv2")
        self.drop_path = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1)) if drop_rate > 0 else None

    def build(self, input_shape):
        if self.layer_scale_init_value > 0:
            self.gamma = self.add_weight(shape=[input_shape[-1]],
                                         initializer=initializers.Constant(self.layer_scale_init_value),
                                         trainable=True,
                                         dtype=tf.float32,
                                         name="gamma")
        else:
            self.gamma = None

    def call(self, x, training=False):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x, training=training)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        return shortcut + x


class Stem(layers.Layer):  # 融合第三个维度,(None, [20, 9, 21] --> (None, [20, 9, 7]

    def __init__(self, dim, name: str = None):  # dim[0] = 32
        super().__init__(name=name)
        self.conv = layers.Conv2D(dim,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  # padding="same",
                                  kernel_initializer=KERNEL_INITIALIZER,
                                  bias_initializer=BIAS_INITIALIZER,
                                  name="conv2d")
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x, training=training)
        return x


class DownSample(layers.Layer):
    def __init__(self, dim, name: str = None):
        super().__init__(name=name)
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.conv = layers.Conv2D(dim,
                                  kernel_size=(2, 3),
                                  strides=(2, 3),
                                  # padding="same",
                                  kernel_initializer=KERNEL_INITIALIZER,
                                  bias_initializer=BIAS_INITIALIZER,
                                  name="conv2d")

    def call(self, x, training=False):
        x = self.norm(x, training=training)
        x = self.conv(x)
        return x


def tt():
    attention = Attention()
    train_ds = generate_MHC_ms("ms_{}_x.npy".format(0), "ms_{}_y.npy".format(0), batch_size=700)
    a = np.zeros([11, 20, 9, 21])
    for images, labels in train_ds:
        print(images.shape)
        temp = attention.predict(images)
        print(temp[1])
        temp = temp * images
        # ans = temp2 * images
        print(np.array(temp).shape)
        break


class ConvNeXt_attention(Model):

    def __init__(self, num_classes: int, depths: list, dims: list, drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        cur = 0
        dp_rates = np.linspace(start=0, stop=drop_path_rate, num=sum(depths))
        self.stage1 = [Block(dim=dims[0],
                             drop_rate=dp_rates[cur + i],
                             layer_scale_init_value=layer_scale_init_value,
                             name=f"stage1_block{i}")
                       for i in range(depths[0])]
        cur += depths[0]

        self.downsample1 = DownSample(dims[1], name="downsample1")

        self.stage2 = [Block(dim=dims[1],
                             drop_rate=dp_rates[cur + i],
                             layer_scale_init_value=layer_scale_init_value,
                             name=f"stage2_block{i}")
                       for i in range(depths[1])]
        self.downsample2 = DownSample(dims[2], name="downsample2")

        cur += depths[1]

        # self.maxpool = layers.MaxPool2D(pool_size=(2, 1), strides=2)
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.head = layers.Dense(units=num_classes,
                                 kernel_initializer=KERNEL_INITIALIZER,
                                 bias_initializer=BIAS_INITIALIZER,
                                 name="head",
                                 activation='Softmax')
        self.attention = Attention()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x, training=False):
        # x = self.stem(x, training=training)  # [20, 9, 21] -->  [20, 9, 7]   dim[0] = 7 ,
        att = self.attention(x)
        x = att * x
        for block in self.stage1:
            x = block(x, training=training)  # 2*3 block

        x = self.downsample1(x, training=training)  # down sample ,[20, 9, 7] --> [10, 3, 42]   dim[1]=42
        for block in self.stage2:
            x = block(x, training=training)  # 2*3 block

        x = self.downsample2(x, training=training)  # down sample ,[10, 3, 42] --> [5, 1, 252]   dim[2]=252

        x = tf.reduce_mean(x, axis=[1, 2])
        # x = self.flatten(x)
        x = self.norm(x, training=training)
        x = self.head(x)
        return x


# [20, 9, 21]
# [2*2*5, 3*3*3 , 21]


def convnext_10_10(num_classes: int):
    model = ConvNeXt_attention(depths=[10, 10],
                               dims=[21, 14, 28],
                               num_classes=num_classes)
    return model
