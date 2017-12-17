"""
CycleGAN模型
"""

from GANs.CycleGAN.modules import generator, discriminator
from GANs.CycleGAN.utils import ImageGenerator, visual_grid, ImagePool
from GANs.CycleGAN.layers import *
from GANs.CycleGAN.reader import Reader
from scipy.misc import imsave, imread, imresize
import tensorflow as tf
import numpy as np
import time
import os
import glob


class CycleGAN(object):
    def __init__(self, args):
        """
        创建整个模型
        :param args:
        """
        self.sess = tf.Session()
        self.args = args
        self._create_placeholders()
        self._create_GANs()
        self._create_losses()
        self._collect_vars()
        self._create_opts()
        self._create_summaries()

    def _create_placeholders(self):
        """
        创建占位符
        :return:
        """
        self.realA_ph = Reader(os.path.join(self.args.datadir, "trainA.tfrecords"), name="real_a").feed()
        # tf.placeholder(tf.float32, shape=[None, self.args.imsize, self.args.imsize, 3], name="real_a")

        self.realB_ph = Reader(os.path.join(self.args.datadir, "trainB.tfrecords"), name="real_b").feed()
        # tf.placeholder(tf.float32, shape=[None, self.args.imsize, self.args.imsize, 3], name="real_b")

        self.fakeA_ph = tf.placeholder(tf.float32, shape=[None, self.args.imsize, self.args.imsize, 3], name="fake_a_sample")
        self.fakeB_ph = tf.placeholder(tf.float32, shape=[None, self.args.imsize, self.args.imsize, 3], name="fake_b_sample")

    def _create_GANs(self):
        """
        创建生成器、判别器
        :return:
        """
        self.fakeA = generator(self.realB_ph, reuse=False, name="Gb2a")
        self.fakeB = generator(self.realA_ph, reuse=False, name="Ga2b")

        self.cycA = generator(self.fakeB, reuse=True, name="Gb2a")
        self.cycB = generator(self.fakeA, reuse=True, name="Ga2b")

        self.Da_fake = discriminator(self.fakeA, reuse=False, name="Da")
        self.Db_fake = discriminator(self.fakeB, reuse=False, name="Db")

        self.Da_real = discriminator(self.realA_ph, reuse=True, name="Da")
        self.Db_real = discriminator(self.realB_ph, reuse=True, name="Db")

        self.Da_fake_sample = discriminator(self.fakeA_ph, reuse=True, name="Da")
        self.Db_fake_sample = discriminator(self.fakeB_ph, reuse=True, name="Db")
        pass

    def _create_losses(self):
        """
        创建损失函数
        :return:
        """
        self.lossCycA = abs_criterion(self.realA_ph, self.cycA, name="Loss_Cyc_A")
        self.lossCycB = abs_criterion(self.realB_ph, self.cycB, name="Loss_Cyc_B")

        with tf.variable_scope("Loss_A2B"):
            self.lossGa2b = tf.add(mae_criterion(self.Db_fake, tf.ones_like(self.Db_fake, name="soft_ones_a2b"), name="Loss_Ga2b"),
                                   tf.add(self.args.clambda * self.lossCycA,
                                          self.args.clambda * self.lossCycB,
                                          name="Cycle_Loss"),
                                   name="Total_Loss")

        with tf.variable_scope("Loss_B2A"):
            self.lossGb2a = tf.add(mae_criterion(self.Da_fake, tf.ones_like(self.Da_fake, name="soft_ones_b2a"), name="Loss_Gb2a"),
                                   tf.add(self.args.clambda * self.lossCycA,
                                          self.args.clambda * self.lossCycB,
                                          name="Cycle_Loss"),
                                   name="Total_Loss")

        self.lossDa_real = mae_criterion(self.Da_real, tf.ones_like(self.Da_real, name="soft_ones_A"), name="Loss_Da_real")
        self.lossDb_real = mae_criterion(self.Db_real, tf.ones_like(self.Db_real, name="soft_ones_B"), name="Loss_Db_real")

        self.lossDa_fake = mae_criterion(self.Da_fake_sample, tf.zeros_like(self.Da_fake_sample, name="soft_zeros_A"), name="Loss_Da_fake")
        self.lossDb_fake = mae_criterion(self.Db_fake_sample, tf.zeros_like(self.Db_fake_sample, name="soft_zeros_B"), name="Loss_Db_fake")

        with tf.variable_scope("Loss_Da"):
            self.lossDa = (self.lossDa_real + self.lossDa_fake) * 0.5
        with tf.variable_scope("Loss_Db"):
            self.lossDb = (self.lossDb_real + self.lossDb_fake) * 0.5
        pass

    def _collect_vars(self):
        """
        搜集可训练变量
        :return:
        """

        self.db_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Db")
        self.da_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Da")

        self.g_vars_a2b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Ga2b")
        self.g_vars_b2a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Gb2a")

        self.db_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Db")
        self.da_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Da")

        self.g_a2b_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Ga2b")
        self.g_b2a_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Gb2a")

        # for i in self.g_b2a_update_ops:
        #     print(i)
        # exit()

        # t_vars = tf.trainable_variables()
        # self.db_vars_ = [var for var in t_vars if 'Db' in var.name]
        # self.da_vars_ = [var for var in t_vars if 'Da' in var.name]
        # self.g_vars_a2b_ = [var for var in t_vars if 'Ga2b' in var.name]
        # self.g_vars_b2a_ = [var for var in t_vars if 'Gb2a' in var.name]

        pass

    def _create_summaries(self):
        """
        创建TensorBoard可记录量
        :return:
        """
        with tf.name_scope("Ga2b_summaries"):
            self.g_a2b_loss_sum = tf.summary.scalar("g_loss_a2b", self.lossGa2b)
            self.g_a2b_lr_sum = tf.summary.scalar("g_a2b_lr", self.g_a2b_lr)
            self.g_a2b_sum = tf.summary.merge([self.g_a2b_loss_sum, self.g_a2b_lr_sum])

        with tf.name_scope("Gb2a_summaries"):
            self.g_b2a_loss_sum = tf.summary.scalar("g_loss_b2a", self.lossGb2a)
            self.g_b2a_lr_sum = tf.summary.scalar("g_b2a_lr", self.g_b2a_lr)
            self.g_b2a_sum = tf.summary.merge([self.g_b2a_loss_sum, self.g_b2a_lr_sum])

        with tf.name_scope("Da_summaries"):
            self.da_loss_sum = tf.summary.scalar("da_loss", self.lossDa)

            self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.lossDa_real)
            self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.lossDa_fake)

            self.da_lr_sum = tf.summary.scalar("da_lr", self.da_lr)

            self.da_sum = tf.summary.merge([self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum, self.da_lr_sum])

        with tf.name_scope("Db_summaries"):
            self.db_loss_sum = tf.summary.scalar("db_loss", self.lossDb)

            self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.lossDb_real)
            self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.lossDb_fake)

            self.db_lr_sum = tf.summary.scalar("db_lr", self.db_lr)
            self.db_sum = tf.summary.merge([self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum, self.db_lr_sum])

        with tf.name_scope("image_summaries"):

            # self.p_ing = tf.placeholder(tf.float32, shape=[1, self.args.imsize * 6, self.args.imsize * 4, 3])
            # self.img_op = tf.summary.image("sample", self.p_ing)

            # real_A, fake_A, real_B, fake_B, cyc_A, cyc_B
            with tf.name_scope("A2B"):
                self.real_A = tf.placeholder(tf.float32, shape=[1, self.args.imsize, self.args.imsize, 3], name="real_A_ph")
                self.fake_B = tf.placeholder(tf.float32, shape=[1, self.args.imsize, self.args.imsize, 3], name="fake_B_ph")
                self.cyc_A = tf.placeholder(tf.float32, shape=[1, self.args.imsize, self.args.imsize, 3], name="cyc_A_ph")

                self.real_A_sum = tf.summary.image("1_real_A", self.real_A)
                self.fake_B_sum = tf.summary.image("2_fake_B", self.fake_B)
                self.cyc_A_sum = tf.summary.image("3_cyc_A", self.cyc_A)

            with tf.name_scope("B2A"):
                self.real_B = tf.placeholder(tf.float32, shape=[1, self.args.imsize, self.args.imsize, 3], name="real_B_ph")
                self.fake_A = tf.placeholder(tf.float32, shape=[1, self.args.imsize, self.args.imsize, 3], name="fake_A_ph")
                self.cyc_B = tf.placeholder(tf.float32, shape=[1, self.args.imsize, self.args.imsize, 3], name="cyc_B_ph")

                self.real_B_sum = tf.summary.image("1_real_B", self.real_B)
                self.fake_A_sum = tf.summary.image("2_fake_A", self.fake_A)
                self.cyc_B_sum = tf.summary.image("3_cyc_B", self.cyc_B)

            self.img_op = tf.summary.merge([self.real_A_sum, self.fake_B_sum, self.cyc_A_sum, self.real_B_sum, self.fake_A_sum, self.cyc_B_sum])
            self.saver = tf.train.Saver()

        pass

    def _create_opts(self):
        """
        创建优化目标
        :return:
        """
        @namespace("Adam")
        def make_optimizer(loss,  lr, var_list):
            """
            创建优化器
            :param loss: 目标loss
            :param lr: 学习率
            :param var_list: 待更新的变量
            :return:
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = lr
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000

            learning_rate = tf.where(tf.greater_equal(global_step, start_decay_step),
                                     tf.train.polynomial_decay(starter_learning_rate,
                                                               global_step - start_decay_step,
                                                               decay_steps,
                                                               end_learning_rate=end_learning_rate,
                                                               power=1.0),
                                     y=starter_learning_rate)

            learning_step = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss,
                                                                                      global_step=global_step,
                                                                                      var_list=var_list)

            return learning_rate, learning_step

        with tf.control_dependencies(self.da_update_ops):
            self.da_lr, self.da_optim = make_optimizer(self.lossDa, self.args.lr_d, var_list=self.da_vars, name="Adam_Da")

        with tf.control_dependencies(self.db_update_ops):
            self.db_lr, self.db_optim = make_optimizer(self.lossDb, self.args.lr_d, var_list=self.db_vars, name="Adam_Db")

        with tf.control_dependencies(self.g_a2b_update_ops):
            self.g_a2b_lr, self.g_a2b_optim = make_optimizer(self.lossGa2b, self.args.lr_g, var_list=self.g_vars_a2b, name="Adam_g_a2b")

        with tf.control_dependencies(self.g_b2a_update_ops):
            self.g_b2a_lr, self.g_b2a_optim = make_optimizer(self.lossGb2a, self.args.lr_g, var_list=self.g_vars_b2a, name="Adam_g_b2a")

        pass

    def train(self):
        """
        训练
        :return:
        """
        # self._create_opts()
        self.writer = tf.summary.FileWriter(self.args.logdir, graph=self.sess.graph, max_queue=1)
        self.sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        start_time = time.time()
        counter = 0

        if self.load(self.args.checkpointdir):
            print(" [*] Load SUCCESS")

        else:
            print(" [!] load failed...")

        # realAtrain = ImageGenerator(os.path.join(self.args.datadir, "trainA"),
        #                             batch_size=1,
        #                             resize=(self.args.imsize, self.args.imsize),
        #                             value_model="tanh",
        #                             rand=True)
        #
        # realBtrain = ImageGenerator(os.path.join(self.args.datadir, "trainB"),
        #                             batch_size=1,
        #                             resize=(self.args.imsize, self.args.imsize),
        #                             value_model="tanh",
        #                             rand=True)

        try:
            fakeApool = ImagePool(self.args.pool_size)
            fakeBpool = ImagePool(self.args.pool_size)

            while not coord.should_stop():

                # Forward G network
                fake_A, fake_B = self.sess.run([self.fakeA, self.fakeB])

                # Update Db network
                _, lossDb, summary_str = self.sess.run([self.db_optim, self.lossDb, self.db_sum],
                                                       feed_dict={self.fakeB_ph: fakeBpool.query(fake_B)})
                self.writer.add_summary(summary_str, counter)

                # Update Da network
                _, lossDa, summary_str = self.sess.run([self.da_optim, self.lossDa, self.da_sum],
                                                       feed_dict={self.fakeA_ph: fakeApool.query(fake_A)})
                self.writer.add_summary(summary_str, counter)

                # Update Ga2b network
                _, lossGa2b, summary_str = self.sess.run([self.g_a2b_optim, self.lossGa2b, self.g_a2b_sum])
                self.writer.add_summary(summary_str, counter)

                # Update Gb2a network
                _, lossGb2a, summary_str = self.sess.run([self.g_b2a_optim, self.lossGb2a, self.g_b2a_sum])
                self.writer.add_summary(summary_str, counter)

                print("step: [%2d] time: %4.4f, lossDa: %.8f, lossDb: %.8f, lossGa2b: %.8f, lossGb2a: %.8f" % (counter,
                                                                                                               time.time() - start_time,
                                                                                                               lossDa,
                                                                                                               lossDb,
                                                                                                               lossGa2b,
                                                                                                               lossGb2a))

                counter += 1
                start_time = time.time()
                if counter % self.args.sample_freq == 1:
                    self.sample_model(self.args.sampledir, counter)

                if counter % 1000 == 2:
                    self.save(self.args.checkpointdir, counter)

        except KeyboardInterrupt:
            print("Interrupted")
            coord.request_stop()

        except Exception as e:
            coord.request_stop(e)

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)
        pass

    def sample_model(self, path, count):
        """
        保存样例
        :param path: 路径
        :param count: 总计数
        :return:
        """
        # realAtest = ImageGenerator(os.path.join(self.args.datadir, "testA"),
        #                            batch_size=4,
        #                            resize=(self.args.imsize, self.args.imsize),
        #                            value_model="tanh",
        #                            rand=True)
        #
        # realBtest = ImageGenerator(os.path.join(self.args.datadir, "testB"),
        #                            batch_size=4,.
        #                            resize=(self.args.imsize, self.args.imsize),
        #                            value_model="tanh",
        #                            rand=True)
        #
        # real_A = next(realAtest)
        # real_B = next(realBtest)

        real_A, fake_B, cyc_A, real_B, fake_A, cyc_B = self.sess.run([self.realA_ph, self.fakeB, self.cycA, self.realB_ph, self.fakeA, self.cycB])

        img = visual_grid(np.concatenate([real_A, fake_B, cyc_A, real_B, fake_A, cyc_B], axis=0), shape=[3, 2])

        if self.args.sample_to_file:
            imsave(os.path.join(path, '%04d.png' % count), img, 'png')

        # img = img[None, :, :, :]
        s_img = self.sess.run(self.img_op, feed_dict={self.real_A: real_A,
                                                      self.fake_B: fake_B,
                                                      self.cyc_A: cyc_A,
                                                      self.real_B: real_B,
                                                      self.fake_A: fake_A,
                                                      self.cyc_B: cyc_B})

        self.writer.add_summary(s_img, global_step=count)
        pass

    def save(self, path, count):
        """
        保存checkpoint
        :param path: 路径
        :param count: 计数
        :return:
        """
        checkpoint_name = "CycleGAN.model"
        self.saver.save(self.sess, os.path.join(path, checkpoint_name), global_step=count)
        pass

    def load(self, path):
        """
        加载checkpoint
        :param path: 路径
        :return:
        """
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(path, checkpoint_name))
            return True
        else:
            return False
        pass

    def test(self, direction):
        """
        测试
        :param direction: A2B、B2A
        :return:
        """
        if self.load(self.args.checkpiontdir):
            print(" [*] Load SUCCESS")

        else:
            print(" [!] load failed...")

        save_path = self.args.sampledir
        for file in glob.glob(os.path.join(self.args.datadir, "*jpg")):
            filename = os.path.split(file)[-1]
            data = imread(file)
            data = imresize(data, size=[self.args.imsize, self.args.imsize])
            data = np.array(data)[None, :, :, :]

            if direction == "A2B":
                ret = self.sess.run(self.fakeB, feed_dict={self.realA_ph: data})

            elif direction == "B2A":
                ret = self.sess.run(self.fakeA, feed_dict={self.realB_ph: data})

            imsave(os.path.join(save_path, filename), ret[0], "png")
        pass