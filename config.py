"""
参数和配置信息
"""

import argparse
from collections import OrderedDict


class Config(object):
    def __init__(self):
        self.name = "CycleGAN model in tf_tutorial"

        self.args = OrderedDict(datadir=("./TFRecords/summer2winter_yosemite", "path of datasets"),
                                imsize=(256, "image size"),
                                mode=('train', 'train or A2B or B2A'),
                                nb_epoch=(200, 'number of epochs'),
                                nb_batch=(1000, 'number of batches in a signle epoch'),
                                clambda=(10.0, 'weight of cyclic loss'),
                                lr_g=(2e-4, 'learning rate of G'),
                                lr_d=(1e-4, 'learning rate of D'),
                                pool_size=(50, 'size of image pool that using by training discriminator'),
                                num_resblock=(9, 'size of resblocks in generator'),
                                sample_freq=(100, 'frequency of updating sample images'),
                                sample_to_file=(True, 'save samples to image file'),
                                logdir=('logs', 'path to save logs'),
                                sampledir=('results/summer2winter_yosemite', 'path to save examples'),
                                checkpointdir=('checkpoint', 'path to save checkpoints')
                                )

    def __call__(self):
        parser = argparse.ArgumentParser(prog=self.name)
        for key, value in self.args.items():
            var, doc = value
            parser.add_argument("--%s" % key, dest=key, type=type(var), default=var, help=doc)

        return parser.parse_args()


if __name__ == "__main__":
    config = Config()
    config = config()
    print("............")
    print(config.clambda)