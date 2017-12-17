"""
构建数据
"""

import tensorflow as tf
import glob
import random
import os


def data_reader(input_dir, shuffle=True):
    """
    Read images from input_dir then shuffle them
    :param input_dir: path of input dir
    :param shuffle: list of string
    :return:
    """
    image_list = glob.glob(os.path.join(input_dir, "*.jpg"))

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee random ordering of images with respect to label in the saved TFRecord
        # files. Make the randomization repeatable
        random.seed(12345)
        random.shuffle(image_list)

    return image_list
    pass


def _int64_feature(value):
    """
    Wrapper for inserting int64 features into Example proto.
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    pass


def _bytes_feature(value):
    """
    Wrapper for inserting bytes features into Example proto.
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
    """
    Build an Example proto for an example.
    :param file_path: path to an image file
    :param image_buffer: JPG encoding of RGB image
    :return:
    """
    file_name = file_path.split("/")[-1]
    example = tf.train.Example(features=tf.train.Features(feature={
        "image/file_name": _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
        "image/encoded_image": _bytes_feature(image_buffer)
    }))
    return example


def data_writer(input_dir, output_file):
    """
    Write data to TFRecords
    :param input_dir: path of input dir
    :param output_file: path of output dir
    :return:
    """
    image_list = data_reader(input_dir)
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dump to TFRecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for i, image_path in enumerate(image_list):

        with tf.gfile.FastGFile(image_path, "rb") as f:
            image_data = f.read()

        example = _convert_to_example(image_path, image_data)
        writer.write(example.SerializeToString())

        if i % 500 == 0:
            print("Processed {}/{}.".format(i, len(image_list)))
    print("Done.")
    writer.close()


if __name__ == "__main__":

    print("Convert X data to TFRecords...")
    data_writer("./datasets/summer2winter_yosemite/trainA", "TFRecords/summer2winter_yosemite/trainA.tfrecords")

    print("Convert Y data to tfrecords...")
    data_writer("./datasets/summer2winter_yosemite/trainB", "TFRecords/summer2winter_yosemite/trainB.tfrecords")