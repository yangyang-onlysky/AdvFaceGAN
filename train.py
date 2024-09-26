# -*- coding: utf-8 -*-
import argparse
import configparser

from AdvFaceGANAttack import *


def main(args):
    print("-------------------------Load configuration-------------------------")
    config_file = args.config
    print("config file：" + config_file)
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')

    if args.tms is not None:
        config.set('Train', 'train_model_name_list', args.tms)
    if args.pert is not None:
        config.set('Train', 'MAX_PERTURBATION', str(args.pert))
    if args.output is not None:
        config.set('Save', 'save_dir', args.output)
    if args.stlossfactor is not None:
        config.set('Train', 'st_loss_factor', str(args.stlossfactor))

    if args.maxssim is not None:
        config.set('Train', 'MAX_SSIM', str(args.maxssim))

    print("------------------------Initialization model-------------------------")
    model = AdvFaceGANAttack(config)

    # start training
    model.start_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Relative path of the configuration file', type=str, default='config/target.ini')
    parser.add_argument('--tms', help='All white box models used for training', type=str, default=None)
    parser.add_argument('--pert', help='Upper limit of perturbation', type=float, default=None)
    parser.add_argument('--output', help='Training results save directory', type=str, default=None)
    parser.add_argument('--stlossfactor', help="stloss's factor'", type=float, default=None)
    parser.add_argument('--maxssim', help='Upper limit of ssim', type=float, default=None)

    args = parser.parse_args()
    main(args)
