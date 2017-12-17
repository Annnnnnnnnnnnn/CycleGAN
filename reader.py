"""
数据解析
"""

import tensorflow as tf
from GANs.CycleGAN.utils import convert2float


class Reader(object):
    def __init__(self, tfrecords_file, image_size=256, min_queue_examples=1000, batch_size=1, num_threads=8, name=""):
        """

        :param tfrecords_file: tfrecords file path
        :param image_size:
        :param min_queue_examples: minimum number of samples to retain in the queue that provides of batches of examples
        :param batch_size: number of images per batch
        :param num_threads: number of preprocess threads
        """
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.name = name
        self.reader = tf.TFRecordReader()

    def feed(self):
        """

        :return: images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            file_name_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(file_name_queue)
            features = tf.parse_single_example(serialized_example, features={
                "image/file_name": tf.FixedLenFeature([], tf.string),
                "image/encoded_image": tf.FixedLenFeature([], tf.string)
            })

            image_buffer = features["image/encoded_image"]
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = self._preprocess(image)
            images = tf.train.shuffle_batch([image],
                                            batch_size=self.batch_size,
                                            capacity=self.min_queue_examples + 3 * self.batch_size,
                                            min_after_dequeue=self.min_queue_examples,
                                            num_threads=self.num_threads)

        return images

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = convert2float(image)
        image.set_shape([self.image_size, self.image_size, 3])
        return image
        pass


def test_reader():
    TRAIN_FILE_1 = "./TFRecords/trainA.tfrecords"
    TRAIN_FILE_2 = "./TFRecords/trainB.tfrecords"

    with tf.Session() as sess:
        reader1 = Reader(TRAIN_FILE_1, batch_size=2)
        reader2 = Reader(TRAIN_FILE_2, batch_size=2)
        images_op1 = reader1.feed()
        images_op2 = reader2.feed()
        # tf.local_variables_initializer().run(session=sess)
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                batch_images1, batch_images2 = sess.run([images_op1, images_op2])
                print("image shape: {}".format(batch_images1))
                print("image shape: {}".format(batch_images2))
                print("="*10)
                step += 1

        except KeyboardInterrupt:
            print("Interrupted")
            coord.request_stop()

        except Exception as e:
            coord.request_stop(e)

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    test_reader()