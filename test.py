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
    model.start_testing()
    # Generate a adversary face
    # model.generate_fake(source_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Aaron_Peirsol\Aaron_Peirsol_0003.jpg",
    #                     target_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Aicha_El_Ouafi\Aicha_El_Ouafi_0003.jpg")
    # model.generate_fake(source_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Gordon_McDonald\Gordon_McDonald_0001.jpg",
    #                     target_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Jake_Gyllenhaal\Jake_Gyllenhaal_0005.jpg")
    # model.generate_fake(source_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Serena_Williams\Serena_Williams_0031.jpg",
    #                     target_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Ana_Claudia_Talancon\Ana_Claudia_Talancon_0001.jpg")
    # model.generate_fake(source_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Tracy_McGrady\Tracy_McGrady_0001.jpg",
    #                     target_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Conrad_Black\Conrad_Black_0001.jpg")
    # model.generate_fake(source_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Yoriko_Kawaguchi\Yoriko_Kawaguchi_0009.jpg",
    #                     target_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Winona_Ryder\Winona_Ryder_0016.jpg")
    # model.generate_fake(source_img_path=r"C:\yy\datasets\celeba-hq\celeba_hq-112x112\00005\00005.jpg",
    #                     target_img_path=r"C:\yy\datasets\celeba-hq\celeba_hq-112x112\00018\00018.jpg")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Relative path of the configuration file", type=str)
    parser.add_argument('--model_path', help='The path of the model being tested', type=str, default=None)
    parser.add_argument('--epoch', help='epoch', type=int, default=None)
    args = parser.parse_args()
    main(args)