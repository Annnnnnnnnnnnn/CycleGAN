"""
main
"""

from GANs.CycleGAN.config import Config
from GANs.CycleGAN.model import CycleGAN
import os


def main(args=None):
    config = Config()
    args = config()
    if not os.path.isdir(args.checkpointdir):
        os.mkdir(args.checkpointdir)

    if not os.path.isdir(args.sampledir):
        os.mkdir(args.sampledir)

    model = CycleGAN(args)

    if args.mode == "train":
        model.train()

    else:
        model.test(args.mode)


if __name__ == "__main__":
    # import re
    #
    # a = "210.155.150.145"
    # strInfo = re.compile("www.getchu.com")
    #
    # with open("getchu.com.img.urls.txt", "r") as f:
    #     url = f.read()
    #
    # newUrl = strInfo.sub(a, url)
    # print(newUrl)
    # with open("img.urls.txt", "w") as f:
    #     f.write(newUrl)
    # exit()
    main()