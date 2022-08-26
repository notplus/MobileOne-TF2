import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

import numpy as np


def conv_bn(out_channels, kernel_size, stride, padding, groups=1):
    model = Sequential([
        layers.Conv2D(out_channels, kernel_size=kernel_size,
              strides=stride, padding=padding, groups=groups, use_bias=False),
        layers.BatchNormalization(),
    ])

    return model


class DepthWiseConv(layers.Layer):
    def __init__(self, inc, kernel_size, stride=1):
        super().__init__()
        padding = 'same'
        self.conv = conv_bn(inc, kernel_size, stride, padding, inc)

    def call(self, x):
        return self.conv(x)


class PointWiseConv(layers.Layer):
    def __init__(self, outc):
        super().__init__()
        self.conv = conv_bn(outc, 1, 1, 'same')

    def call(self, x):
        return self.conv(x)


class MobileOneBlock(layers.Layer):

    def __init__(self, in_channels, out_channels, k,
                 stride=1, dilation=1, deploy=False):
        super(MobileOneBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = deploy
        kernel_size = 3
        padding = 1
        assert kernel_size == 3
        assert padding == 1
        self.k = k

        self.nonlinearity = layers.ReLU()

        if deploy:
            self.dw_reparam = layers.Conv2D(in_channels, kernel_size=kernel_size, strides=stride,
                                            padding='same', groups=in_channels, dilation_rate=dilation, use_bias=True)
            self.pw_reparam = layers.Conv2D(
                out_channels, kernel_size=1, strides=1, use_bias=True)

        else:
            self.dw_bn_layer = layers.BatchNormalization(
            ) if stride == 1 else None
            for k_idx in range(k):
                setattr(self, f'dw_3x3_{k_idx}',
                        DepthWiseConv(in_channels, 3, stride=stride)
                        )
            self.dw_1x1 = DepthWiseConv(in_channels, 1, stride=stride)

            self.pw_bn_layer = layers.BatchNormalization(
            ) if out_channels == in_channels else None
            for k_idx in range(k):
                setattr(self, f'pw_1x1_{k_idx}',
                        PointWiseConv(out_channels)
                        )

    def call(self, inputs):
        if self.deploy:
            x = self.dw_reparam(inputs)
            x = self.nonlinearity(x)
            x = self.pw_reparam(x)
            x = self.nonlinearity(x)
            return x

        if self.dw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.dw_bn_layer(inputs)

        x_conv_3x3 = []
        for k_idx in range(self.k):
            x = getattr(self, f'dw_3x3_{k_idx}')(inputs)
            x_conv_3x3.append(x)
        x_conv_1x1 = self.dw_1x1(inputs)

        x = id_out + x_conv_1x1 + sum(x_conv_3x3)
        x = self.nonlinearity(x)

        # 1x1 conv
        if self.pw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.pw_bn_layer(x)
        x_conv_1x1 = []
        for k_idx in range(self.k):
            x_conv_1x1.append(getattr(self, f'pw_1x1_{k_idx}')(x))
        x = id_out + sum(x_conv_1x1)
        x = self.nonlinearity(x)
        return x

#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.

    def get_equivalent_kernel_bias(self):
        dw_kernel_3x3 = []
        dw_bias_3x3 = []
        for k_idx in range(self.k):
            k3, b3 = self._fuse_bn_tensor(
                getattr(self, f"dw_3x3_{k_idx}").conv)
            # print(k3.shape, b3.shape)
            dw_kernel_3x3.append(k3)
            dw_bias_3x3.append(b3)
        dw_kernel_1x1, dw_bias_1x1 = self._fuse_bn_tensor(self.dw_1x1.conv)
        dw_kernel_id, dw_bias_id = self._fuse_bn_tensor(
            self.dw_bn_layer, self.in_channels)
        dw_kernel = sum(dw_kernel_3x3) + \
            self._pad_1x1_to_3x3_tensor(dw_kernel_1x1) + dw_kernel_id
        dw_bias = sum(dw_bias_3x3) + dw_bias_1x1 + dw_bias_id
        # pw
        pw_kernel = []
        pw_bias = []
        for k_idx in range(self.k):
            k1, b1 = self._fuse_bn_tensor(
                getattr(self, f"pw_1x1_{k_idx}").conv)
            # print(k1.shape)
            pw_kernel.append(k1)
            pw_bias.append(b1)
        pw_kernel_id, pw_bias_id = self._fuse_bn_tensor(self.pw_bn_layer, 1)

        pw_kernel_1x1 = sum(pw_kernel) + pw_kernel_id
        pw_bias_1x1 = sum(pw_bias) + pw_bias_id
        return dw_kernel, dw_bias, pw_kernel_1x1, pw_bias_1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            if self.stride == 2:
                return tf.pad(kernel1x1, tf.constant([[0, 2], [0, 2], [0, 0], [0, 0]]), "CONSTANT")
            else:
                return tf.pad(kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]]), "CONSTANT")

    def _fuse_bn_tensor(self, branch, groups=None):
        if branch is None:
            return 0, 0
        if isinstance(branch, Sequential):
            kernel = branch.layers[0].kernel
            running_mean = branch.layers[1].moving_mean
            running_var = branch.layers[1].moving_variance
            gamma = branch.layers[1].gamma
            beta = branch.layers[1].beta
            eps = branch.layers[1].epsilon

        else:
            assert isinstance(branch, layers.BatchNormalization)
            input_dim = self.in_channels // groups  # self.groups
            if groups == 1:
                ks = 1
            else:
                ks = 3
            kernel_value = np.zeros(
                (ks, ks, input_dim, self.in_channels), dtype=np.float32)
            for i in range(self.in_channels):
                if ks == 1:
                    kernel_value[0, 0, i % input_dim, i] = 1
                else:
                    kernel_value[1, 1, i % input_dim, i] = 1
            self.id_tensor = tf.convert_to_tensor(kernel_value)

            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        std = tf.sqrt((running_var + eps))
        t = tf.reshape((gamma / std), (1, 1, 1, -1))
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        dw_kernel, dw_bias, pw_kernel, pw_bias = self.get_equivalent_kernel_bias()

        self.dw_reparam = layers.Conv2D(
            filters=self.dw_3x3_0.conv.layers[0].filters,
            kernel_size=self.dw_3x3_0.conv.layers[0].kernel_size,
            strides=self.dw_3x3_0.conv.layers[0].strides,
            padding=self.dw_3x3_0.conv.layers[0].padding,
            groups=self.dw_3x3_0.conv.layers[0].groups,
            use_bias=True,
            weights=[dw_kernel.numpy(), dw_bias.numpy()]
        )
        self.pw_reparam = layers.Conv2D(
            filters=self.pw_1x1_0.conv.layers[0].filters,
            kernel_size=1,
            strides=1,
            use_bias=True
        )

        self.dw_reparam.build((1, 32, 32, self.dw_3x3_0.conv.layers[0].groups))
        self.pw_reparam.build((1, 32, 32, self.dw_3x3_0.conv.layers[0].groups))
        self.dw_reparam.set_weights([dw_kernel, dw_bias])
        self.pw_reparam.set_weights([pw_kernel, pw_bias])

        # for para in self.parameters():
        #     para.detach_()
        self.__delattr__('dw_1x1')
        for k_idx in range(self.k):
            self.__delattr__(f'dw_3x3_{k_idx}')
            self.__delattr__(f'pw_1x1_{k_idx}')
        if hasattr(self, 'dw_bn_layer'):
            self.__delattr__('dw_bn_layer')
        if hasattr(self, 'pw_bn_layer'):
            self.__delattr__('pw_bn_layer')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class MobileOneNet(Model):
    def __init__(self, blocks, ks, channels, strides, width_muls, deploy=False):
        super().__init__()

        self.stage_num = len(blocks)
        self.stage0 = MobileOneBlock(3, int(channels[0] * width_muls[0]), ks[0], stride=strides[0], deploy=deploy)
        # self.stage0 = Sequential([
        #     layers.Conv2D(int(channels[0] * width_muls[0]), 3, 2, 'same', use_bias=False),
        #     layers.BatchNormalization(),
        #     layers.ReLU(),
        # ])

        in_channels = int(channels[0] * width_muls[0])
        for idx, block_num in enumerate(blocks[1:]):
            idx += 1
            module = Sequential()
            out_channels = int(channels[idx] * width_muls[idx])
            for b_idx in range(block_num):
                stride = strides[idx] if b_idx == 0 else 1
                block = MobileOneBlock(
                    in_channels, out_channels, ks[idx], stride, deploy=deploy)
                in_channels = out_channels
                module.add(block)
            setattr(self, f"stage{idx}", module)

    def call(self, inputs):
        x0 = self.stage0(inputs)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return x5


def make_mobileone_s0(width_mult=1, deploy=False):
    blocks = [
        1, 2, 8, 5, 5, 1
    ]
    strides = [
        2, 2, 2, 2, 1, 2
    ]
    ks = [
        4, 4, 4, 4, 4, 4
    ] if deploy is False else \
        [
            1, 1, 1, 1, 1, 1
    ]
    width_muls = [
        0.75 * width_mult, 0.75 * width_mult, 1 * width_mult, 1 *
        width_mult, 1 * width_mult, 2 * width_mult
    ]  # 261 M flops
    channels = [
        64, 64, 128, 256, 256, 512, 512
    ]

    model = MobileOneNet(blocks, ks, channels, strides,
                         width_muls, deploy)

    return model

