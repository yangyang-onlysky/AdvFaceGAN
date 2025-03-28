# -*- coding: utf-8 -*-
import argparse
import configparser
import time

from AdvFaceGANAttack import *
from utils.dataset import *


def main(args):
    print("-------------------------Load configuration-------------------------")
    config_file = args.config
    print("config file：" + config_file)
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')

    if args.model_path is not None:
        config.set('Test', 'test_model_dir', args.model_path)
    if args.epoch is not None:
        config.set('Test', 'test_epoch_id', str(args.epoch))

    # Load the pre-trained model
    print("-------------------------Initialization model------------------------")
    model = AdvFaceGANAttack(config)
    model.eval()

    # Test the entire data set
    # model.start_testing()

    # Generate a adversary face
    model.generate_fake(source_img_path=r"C:\yy\datasets\505\505-aligned-112x112\gj\1.png",
                        target_img_path=r"C:\yy\datasets\505\505-aligned-112x112\yy\1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Relative path of the configuration file", type=str)
    parser.add_argument('--model_path', help='The path of the model being tested', type=str, default=None)
    parser.add_argument('--epoch', help='epoch', type=int, default=None)
    args = parser.parse_args()
    main(args)