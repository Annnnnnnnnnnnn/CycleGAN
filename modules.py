"""
模块化的生成器与判别器
"""

from GANs.CycleGAN.layers import *


@namespace("generator")
def generator(x, reuse, gdim=32):
    """
    生成器
    :param x: Tensor
    :param reuse: 是否重用变量
    :param gdim:
    :return: Tensor
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("encoder"):
        # A 7*7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
        x = reflect_pad(x, 3, name="rp_input")
        x = conv2d(x, filters=gdim, kernel=7, strides=1, name="conv_input", padding="VALID")
        x = batch_norm(x, name="bn_input")
        x = tf.nn.relu(x)

        # 256 * 256 * 32
        # A 3*3 Convolution-BatchNorm-ReLU layer with k filter and stride 2
        for i in range(1, 3):
            # x = reflect_pad(x, name="rp_%d" % i)
            x = conv2d(x, filters=gdim * (2 ** i), kernel=3, strides=2, padding="SAME", name="conv_%d" % i)
            x = batch_norm(x, name="bn_%d" % i)
            x = tf.nn.relu(x)
        # res = x

    # 64 * 64 * 128
    # A residual block that contains two 3*3 convolutional layers with the same number of filters on both layer
    with tf.variable_scope("transform"):
        for i in range(9):
            x = res_block(x, filters=gdim * (2 ** 2), name="res%d" % (i + 1))

    # 64 * 64 * 128
    # A 3*3 fractional-stride-Convolution-BatchNorm-ReLU layer with k filters, stride 1/2
    x_shape = x.get_shape().as_list()  # tf.shape(x)
    # print(x_shape, x.get_shape(), tf.shape(x)[0])
    with tf.variable_scope("decoder"):
        # x = res + x
        for i in range(1, 3):
            x = deconv2d(x, out_shape=[x_shape[0], x_shape[1] * (2 ** i), x_shape[1] * (2 ** i), gdim * (2 ** (2 - i))], kernel=3, strides=2, name="convT%d" % i)
            x = batch_norm(x, name="bn_%d" % i)
            x = lrelu(x)

        # conv layer
        # Note: the paper said that ReLU and _norm were used
        # but actually tanh was used and no _norm here
        x = reflect_pad(x, 3, name="rp_output")
        x = conv2d(x, filters=3, kernel=7, padding="VALID", name="conv_out")
        # x = batch_norm(x, name="bn_output")
        x = tf.nn.tanh(x, name="output")

    return x


@namespace("discriminator")
def discriminator(x, reuse, ddim=64):
    """
    判别器
    :param x: Tensor
    :param reuse: 是否重用变量
    :param ddim:
    :return:
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()

    x = conv2d(x, filters=ddim, kernel=4, strides=2, padding="SAME", name="input")
    x = lrelu(x)

    # A 4*4 Convolution-BatchNorm_LeakyReLU layer with k filters and stride 2
    for i in range(1, 4):
        x = conv2d(x, filters=ddim * (2 ** i), kernel=4, strides=2, padding="SAME", name="conv%d" % i)
        x = batch_norm(x, name="bn%d" % i)
        x = lrelu(x)

    # Last convolutional layer of discriminator network (1 filter with size 4*4, stride 1)
    x = conv2d(x, filters=1, kernel=4, strides=1, use_bias=True, padding="SAME", name="output")
    # x = tf.nn.sigmoid(x)

    return x


if __name__ == "__main__":
    a = [[1, 2, 3],
         [3, 4, 5]]
    padding = [[1, 1],
               [2, 2]]
    a = tf.Variable(a, dtype=tf.float32, trainable=False)
    b = tf.pad(a, padding, mode="SYMMETRIC")
    c = tf.Variable(4, dtype=tf.float32, trainable=False)
    op = tf.assign(c, 5)
    print(op)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(a.eval())
        print(b.eval())
        print(op.eval())