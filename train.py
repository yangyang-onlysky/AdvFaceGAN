# -*- coding: utf-8 -*-
import argparse
import configparser

from AdvFaceGANAttack import *


def main(args):
    print("-------------------------加载配置-------------------------")
    config_file = args.config
    print("配置文件：" + config_file)
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

    print("------------------------初始化模型-------------------------")
    model = AdvFaceGANAttack(config)

    # 启动训练
    model.start_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='基础配置文件', type=str, default='config/target.ini')
    parser.add_argument('--tms', help='训练所用所有白盒模型', type=str, default=None)
    parser.add_argument('--pert', help='扰动下限', type=float, default=None)
    parser.add_argument('--output', help='训练成果保存目录', type=str, default=None)
    parser.add_argument('--stlossfactor', help='stloss的因子', type=float, default=None)
    args = parser.parse_args()
    main(args)
