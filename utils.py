"""
工具类
"""

from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np
import os
import random
import math
from glob import glob


class ImageGenerator(object):
    def __init__(self, root: str, batch_size=1, resize: tuple=None, value_model: str="origin", rand=False):
        """
        图片获取生成器, 随机获取图片数据
        :param root: 遍历目录
        :param batch_size: 分组数量
        :param resize: 更改大小（None为不）
        :param value_model: 结果模式（origin: 原样，sigmoid: 0~1区间，tanh: -1~1区间）
        """
        self._root = root
        self._batch_size = batch_size
        self._resize = resize
        self._value_model = value_model
        self._rand = rand
        self._img_list = np.array(glob(os.path.join(root, "*.jpg")))
        self._len = math.ceil(len(self._img_list) / self._batch_size)

        np.random.shuffle(self._img_list)

        if self._batch_size > 0:
            self._batch_idx = np.array_split(np.arange(len(self._img_list)), self._len)

        else:
            self._batch_idx = [np.arange(self._len)]

        self._batch_id = 0

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._batch_id >= len(self._batch_idx):
            self._batch_id = 0
            if not self._rand:
                raise StopIteration
        out = self._img_list[self._batch_idx[self._batch_id]]
        self._batch_id += 1

        imgs = []
        for filename in out:
            img = imread(filename, mode="RGB")
            if self._resize:
                img = imresize(img, self._resize)
            imgs.append(img)

        imgs = np.array(imgs).astype(np.float32)

        if self._value_model == "origin":
            return imgs

        elif self._value_model == "sigmoid":
            return imgs / 255.

        elif self._value_model == "tanh":
            return (imgs / 127.5) - 1.

        pass


def image_generator(root: str, batch_size=1, resize: tuple=None, value_model: str="origin", rand=True):
    """
    图片获取生成器, 随机获取图片数据
    :param root: 遍历目录
    :param batch_size: 分组数量
    :param resize: 更改大小（None为不）
    :param value_model: 结果模式（origin: 原样，sigmoid: 0~1区间，tanh: -1~1区间）
    :param rand:
    :return: 返回一个迭代器
    """
    img_list = glob(os.path.join(root, "*.jpg"))

    if rand:
        while True:
            imgs = []
            for _ in range(batch_size):
                filename = random.choice(img_list)
                img = imread(filename, mode="RGB")
                if resize:
                    img = imresize(img, resize)
                imgs.append(img)
            imgs = np.array(imgs).astype(np.float32)

            if value_model == "origin":
                yield imgs

            elif value_model == "sigmoid":
                yield imgs / 255.

            elif value_model == "tanh":
                yield (imgs / 127.5) - 1.

    else:
        imgs = []
        for filename in img_list:
            img = imread(filename, mode="RGB")
            if resize:
                img = imresize(img, resize)
            imgs.append(img)

            if len(imgs) == batch_size:
                rt = np.array(imgs).astype(np.float32)
                imgs = []

                if value_model == "origin":
                    yield rt

                elif value_model == "sigmoid":
                    yield rt / 255.

                elif value_model == "tanh":
                    yield (rt / 128.) - 128.


def convert2int(image):
    """
    Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    :param image:
    :return:
    """
    return tf.image.convert_image_dtype((image + 1.0) / 2.0, tf.uint8)


def convert2float(image):
    """
    Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    :param image:
    :return:
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


def visual_grid(X: np.array, shape: tuple((int, int))):
    """
    将X中的图片平铺放入新的numpy.array中，用于可视化
    :param X: 图片集合(numpy.array)
    :param shape: 表格形状（行，列）图片数
    :return: 合成后图片array
    """
    nh, nw = shape
    h, w = X.shape[1:3]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n // nh  # nw
        i = n % nh  # nw
        if n >= nh * nw:
            break
        img[i*h:i*h+h, j*w:j*w+w, :] = x
    return img


def namespace(default_name):
    """
    variable space装饰器
    产生带name的装饰器
    :param default_name: 待装饰函数
    :return:
    """
    def deco(fn):
        def wrapper(*args, **kwargs):
            if "name" in kwargs:
                name = kwargs["name"]
                kwargs.pop("name")

            else:
                name = default_name

            with tf.variable_scope(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco


class ImagePool(object):
    """
    数据池
    用以装载固定量的数据，并提供获取全部及随机获取一个的途径
    """
    def __init__(self, pool_size=50):
        self._images = []
        self.pool_size = pool_size

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self._images) < self.pool_size:
            self._images.append(image)
            return image

        else:
            p = random.random()
            if p > 0.5:
                # 使用历史图片
                random_id = random.randrange(0, self.pool_size, step=1)
                temp = self._images[random_id].copy()
                self._images[random_id] = image
                return temp

            else:
                return image


if __name__ == "__main__":
    a = ImageGenerator("./datasets/vangogh2photo/trainA/", batch_size=2, rand=True)





