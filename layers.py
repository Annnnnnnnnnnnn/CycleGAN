"""
可能用到的层或ops
"""
import tensorflow as tf
from GANs.CycleGAN.utils import namespace


@namespace("instance_norm")
def instance_norm(x):
    """
    单项正则化（Generator中用到）
    :param x: 输入Tensor
    :return: 输出Tensor
    """
    depth = x.get_shape()[-1]
    scale = tf.get_variable(name="scale", shape=[depth], initializer=tf.random_normal_initializer(mean=1., stddev=0.02))
    offset = tf.get_variable(name="offset", shape=[depth], initializer=tf.constant_initializer(0.))
    mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (x - mean) * inv
    return tf.add(tf.multiply(scale, normalized), offset)


@namespace("batch_norm")
def batch_norm(x):
    """
    批数据正则
    此处不进行trainning控制，改而使用变量搜集后，在反向传播时限制训练变量进行控制(Discriminator中用到)
    :param x:
    :return:
    """
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        scale=True,
                                        is_training=True,
                                        updates_collections=tf.GraphKeys.UPDATE_OPS)  # tf.layers.batch_normalization(x, trainable=True)


@namespace("reflect_pad")
def reflect_pad(x, size=1):
    """
    反射式补边
    以边框临近位置的内容进行补边
    :param x: Tensor
    :param size: 边框大小
    :return: Tensor
    """
    return tf.pad(x, paddings=((0, 0), (size, size), (size, size), (0, 0)), mode="REFLECT")


@namespace("conv2d")
def conv2d(x, filters, kernel, strides=1, stddev=0.02, use_bias=False, padding="SAME"):
    """
    卷积层
    :param x: Tensor
    :param filters: filter数量
    :param kernel: kernel大小（正方形kernel）
    :param strides: 步长
    :param stddev: 初始化标准差
    :param use_bias: 是否使用偏置
    :param padding: 补零方式（VALID、SAME）
    :return: Tensor
    """
    w = tf.get_variable("w", shape=[kernel, kernel, x.get_shape()[-1], filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)

    if use_bias:
        b = tf.get_variable("b", shape=[filters], initializer=tf.constant_initializer(0.))
        conv = tf.nn.bias_add(conv, b)

    return conv


@namespace("leaky_relu")
def lrelu(x, leak=0.2):
    """
    Leaky ReLU激活函数
    :param x: Tensor
    :param leak: 负轴的泄露系数
    :return: Tensor
    """
    return tf.maximum(x, tf.multiply(x, leak))


@namespace("res_block")
def res_block(x, filters=128):
    """
    残差块
    :param x: Tensor
    :param filters: filter数量
    :return: Tensor
    """
    y = reflect_pad(x, name="rp1")
    y = conv2d(y, filters=filters, kernel=3, name='conv1', padding="VALID")
    y = batch_norm(y, name="bn_1")
    y = tf.nn.relu(y)

    y = reflect_pad(y, name="rp2")
    y = conv2d(y, filters=filters, kernel=3, name="conv2", padding="VALID")
    y = batch_norm(y, name="bn_2")
    return tf.add(x, y)


@namespace("unsampling")
def unsampling(x):
    """
    二倍上采样
    :param x: Tensor
    :return: Tensor
    """
    input_shape = x.get_shape()
    out_put_shape = (None, input_shape[1] * 2, input_shape[2] * 2, None)
    x = tf.image.resize_bilinear(x, tf.shape(x)[1:3] * 2, name="unsampling")
    x.set_shape(out_put_shape)
    return x


@namespace("conv2dTranspose")
def deconv2d(x, out_shape, kernel, strides=1, stddev=0.02, use_bias=False, padding="SAME"):
    """
    转置卷积
    :param x: Tensor
    :param out_shape: filter数量
    :param kernel: kernel大小（正方形kernel）
    :param strides: 步长
    :param stddev: 初始化标准差
    :param use_bias: 是否使用偏置
    :param padding: 补零方式（VALID、SAME）
    :return: Tensor
    """

    # x = unsampling(x, name="upsample")
    # x = reflect_pad(x, 1, name="rp")
    # x = conv2d(x, filters, 3, name="conv")

    w = tf.get_variable("w", shape=[kernel, kernel, out_shape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=[1, strides, strides, 1], padding=padding)

    if use_bias:
        b = tf.get_variable("b", shape=[out_shape[-1]], initializer=tf.constant_initializer(0.))
        deconv = tf.nn.bias_add(deconv, b)

    return deconv


@namespace("abs_criterion")
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


@namespace("mae_criterion")
def mae_criterion(in_, target):
    return tf.reduce_mean(tf.square(in_ - target))


def batch_norm_(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta',
                           trainable=True)

        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma',
                            trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed