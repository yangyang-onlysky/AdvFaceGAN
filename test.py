# -*- coding: utf-8 -*-
import argparse
import configparser
import time

from AdvFaceGANAttack import *
from utils.dataset import *


def main(args):
    print("-------------------------加载配置-------------------------")
    config_file = args.config
    print("配置文件：" + config_file)
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')

    if args.model_path is not None:
        config.set('Test', 'test_model_dir', args.model_path)
    if args.epoch is not None:
        config.set('Test', 'test_epoch_id', str(args.epoch))

    # 加载预训练模型
    print("-------------------------初始化模型------------------------")
    model = AdvFaceGANAttack(config)
    model.eval()

    # 测试整个数据集
    # model.start_testing()
    # 测试单张
    # model.generate_fake(source_img_path=r"C:\yy\datasets\lfw\lfw-aligned-112x112\Aaron_Peirsol\Aaron_Peirsol_0003.jpg",
    #                     target_img_path=r"C:\yy\yy\source\Desktop\读研是一条艰苦的道路\1. 论文\做实验\myCode\4 FrAdv\AdvFaceGAN\data\celeba_hq-112x112\target\target.jpg")
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
    #
    model.generate_fake(source_img_path=r"C:\yy\source\Desktop\读研是一条艰苦的道路\1. 论文\做实验\myCode\myCode\4 FrAdv\AdvFaceGAN\data\celeba_hq-112x112\WIN_20240809_14_03_26_Pro\1.jpg",
                        target_img_path=r"C:\yy\source\Desktop\读研是一条艰苦的道路\1. 论文\做实验\myCode\myCode\4 FrAdv\AdvFaceGAN\data\celeba_hq-112x112\WIN_20240809_14_03_12_Pro\2.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="配置文件相对路径", type=str)
    parser.add_argument('--model_path', help='测试的模型路径', type=str, default=None)
    parser.add_argument('--epoch', help='测试轮数', type=int, default=None)
    args = parser.parse_args()
    main(args)
'''
lfw:
扰动量3 700轮 集成 ['ArcFace','FaceNet-VGGFace2', 'Mobilenet-stride1','ShuffleNet_V1_GDConv','ResNet50','IR50-CosFace','IR50-ArcFace','IR50-SphereFace']
FaceNet-casia  benchmark rate:0.981000
FaceNet-casia  攻击前 success rate:0.979779
FaceNet-casia  平均 ssim:0.942353
FaceNet-casia  攻击后 success rate:0.240976
--------------------开始测试 ArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:23<00:00,  2.25it/s]
ArcFace  benchmark rate:0.995000
ArcFace  攻击前 success rate:0.996825
ArcFace  平均 ssim:0.942654
ArcFace  攻击后 success rate:0.102941
--------------------开始测试 SphereFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:47<00:00,  3.94it/s]
SphereFace  benchmark rate:0.981833
SphereFace  攻击前 success rate:0.981283
SphereFace  平均 ssim:0.942281
SphereFace  攻击后 success rate:0.122995
--------------------开始测试 CosFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:48<00:00,  3.89it/s]
CosFace  benchmark rate:0.986833
CosFace  攻击前 success rate:0.981451
CosFace  平均 ssim:0.942455
CosFace  攻击后 success rate:0.156417
--------------------开始测试 MobileFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:48<00:00,  3.85it/s]
MobileFace  benchmark rate:0.994500
MobileFace  攻击前 success rate:0.997493
MobileFace  平均 ssim:0.942603
MobileFace  攻击后 success rate:0.063336
--------------------开始测试 IR50-PGDArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:47<00:00,  1.74it/s]
IR50-PGDArcFace  benchmark rate:0.876667
IR50-PGDArcFace  攻击前 success rate:0.905247
IR50-PGDArcFace  平均 ssim:0.942406
IR50-PGDArcFace  攻击后 success rate:0.860963
--------------------开始测试 IR50-TradesArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:56<00:00,  1.61it/s]
IR50-TradesArcFace  benchmark rate:0.950667
IR50-TradesArcFace  攻击前 success rate:0.929144
IR50-TradesArcFace  平均 ssim:0.942444
IR50-TradesArcFace  攻击后 success rate:0.807487
--------------------开始测试 IR50-Softmax-BR---------------------
Load existing checkpoint
100%|██████████| 187/187 [02:04<00:00,  1.50it/s]
IR50-Softmax-BR  benchmark rate:0.996000
IR50-Softmax-BR  攻击前 success rate:0.994652
IR50-Softmax-BR  平均 ssim:0.942509
IR50-Softmax-BR  攻击后 success rate:0.247493
--------------------开始测试 IR50-Softmax-RP---------------------
Load existing checkpoint
100%|██████████| 187/187 [15:36<00:00,  5.01s/it]
IR50-Softmax-RP  benchmark rate:0.994167
IR50-Softmax-RP  攻击前 success rate:0.994820
IR50-Softmax-RP  平均 ssim:0.942374
IR50-Softmax-RP  攻击后 success rate:0.270889
--------------------开始测试 IR50-Softmax-JPEG---------------------
Load existing checkpoint
100%|██████████| 187/187 [02:32<00:00,  1.22it/s]
IR50-Softmax-JPEG  benchmark rate:0.995833
IR50-Softmax-JPEG  攻击前 success rate:0.995989
IR50-Softmax-JPEG  平均 ssim:0.942409
IR50-Softmax-JPEG  攻击后 success rate:0.265207

扰动量4 650轮 集成 ['ArcFace','FaceNet-VGGFace2', 'Mobilenet-stride1','ShuffleNet_V1_GDConv','ResNet50','IR50-CosFace','IR50-ArcFace','IR50-SphereFace']
FaceNet-casia  benchmark rate:0.981000
FaceNet-casia  攻击前 success rate:0.978777
FaceNet-casia  平均 ssim:0.942628
FaceNet-casia  平均 psnr:33.714857
FaceNet-casia  平均 mse:27.643872
FaceNet-casia  攻击后 success rate:0.106785
--------------------开始测试 ArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [03:44<00:00,  1.20s/it]
ArcFace  benchmark rate:0.995000
ArcFace  攻击前 success rate:0.997995
ArcFace  平均 ssim:0.942513
ArcFace  平均 psnr:33.715079
ArcFace  平均 mse:27.642442
ArcFace  攻击后 success rate:0.036765
--------------------开始测试 SphereFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [04:47<00:00,  1.54s/it]
SphereFace  benchmark rate:0.981833
SphereFace  攻击前 success rate:0.984793
SphereFace  平均 ssim:0.942724
SphereFace  平均 psnr:33.713775
SphereFace  平均 mse:27.650742
SphereFace  攻击后 success rate:0.054311
--------------------开始测试 CosFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [02:10<00:00,  1.44it/s]
CosFace  benchmark rate:0.986833
CosFace  攻击前 success rate:0.981618
CosFace  平均 ssim:0.942700
CosFace  平均 psnr:33.714785
CosFace  平均 mse:27.644299
CosFace  攻击后 success rate:0.063670
--------------------开始测试 MobileFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [04:01<00:00,  1.29s/it]
MobileFace  benchmark rate:0.994500
MobileFace  攻击前 success rate:0.997660
MobileFace  平均 ssim:0.942630
MobileFace  平均 psnr:33.713932
MobileFace  平均 mse:27.649719
MobileFace  攻击后 success rate:0.020889
--------------------开始测试 IR50-PGDArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [07:54<00:00,  2.54s/it]
IR50-PGDArcFace  benchmark rate:0.876667
IR50-PGDArcFace  攻击前 success rate:0.909759
IR50-PGDArcFace  平均 ssim:0.942692
IR50-PGDArcFace  平均 psnr:33.715339
IR50-PGDArcFace  平均 mse:27.640763
IR50-PGDArcFace  攻击后 success rate:0.833055
--------------------开始测试 IR50-TradesArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [06:42<00:00,  2.15s/it]
IR50-TradesArcFace  benchmark rate:0.950667
IR50-TradesArcFace  攻击前 success rate:0.929479
IR50-TradesArcFace  平均 ssim:0.942654
IR50-TradesArcFace  平均 psnr:33.715287
IR50-TradesArcFace  平均 mse:27.641125
IR50-TradesArcFace  攻击后 success rate:0.734291
--------------------开始测试 IR50-Softmax-BR---------------------
Load existing checkpoint
100%|██████████| 187/187 [08:27<00:00,  2.71s/it]
IR50-Softmax-BR  benchmark rate:0.996000
IR50-Softmax-BR  攻击前 success rate:0.995488
IR50-Softmax-BR  平均 ssim:0.942836
IR50-Softmax-BR  平均 psnr:33.717212
IR50-Softmax-BR  平均 mse:27.628869
IR50-Softmax-BR  攻击后 success rate:0.110628
--------------------开始测试 IR50-Softmax-RP---------------------
Load existing checkpoint
100%|██████████| 187/187 [2:25:53<00:00, 46.81s/it]
IR50-Softmax-RP  benchmark rate:0.994167
IR50-Softmax-RP  攻击前 success rate:0.994820
IR50-Softmax-RP  平均 ssim:0.942610
IR50-Softmax-RP  平均 psnr:33.712401
IR50-Softmax-RP  平均 mse:27.659478
IR50-Softmax-RP  攻击后 success rate:0.131183

celeba-hq:
扰动量3 700轮 集成 ['ArcFace','FaceNet-VGGFace2', 'Mobilenet-stride1','ShuffleNet_V1_GDConv','ResNet50','IR50-CosFace','IR50-ArcFace','IR50-SphereFace']
--------------------开始测试 FaceNet-casia---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:01<00:00,  3.05it/s]
FaceNet-casia  benchmark rate:0.981000
FaceNet-casia  攻击前 success rate:0.984459
FaceNet-casia  平均 ssim:0.952903
FaceNet-casia  攻击后 success rate:0.304479
--------------------开始测试 ArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:20<00:00,  2.32it/s]
ArcFace  benchmark rate:0.995000
ArcFace  攻击前 success rate:0.996324
ArcFace  平均 ssim:0.953271
ArcFace  攻击后 success rate:0.126337
--------------------开始测试 SphereFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:43<00:00,  4.25it/s]
SphereFace  benchmark rate:0.981833
SphereFace  攻击前 success rate:0.986130
SphereFace  平均 ssim:0.953062
SphereFace  攻击后 success rate:0.163436
--------------------开始测试 CosFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:50<00:00,  3.73it/s]
CosFace  benchmark rate:0.986833
CosFace  攻击前 success rate:0.984793
CosFace  平均 ssim:0.953016
CosFace  攻击后 success rate:0.233456
--------------------开始测试 MobileFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:45<00:00,  4.12it/s]
MobileFace  benchmark rate:0.994500
MobileFace  攻击前 success rate:0.995154
MobileFace  平均 ssim:0.952850
MobileFace  攻击后 success rate:0.071524
--------------------开始测试 IR50-PGDArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:37<00:00,  1.91it/s]
IR50-PGDArcFace  benchmark rate:0.876667
IR50-PGDArcFace  攻击前 success rate:0.823028
IR50-PGDArcFace  平均 ssim:0.952927
IR50-PGDArcFace  攻击后 success rate:0.750668
--------------------开始测试 IR50-TradesArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:41<00:00,  1.85it/s]
IR50-TradesArcFace  benchmark rate:0.950667
IR50-TradesArcFace  攻击前 success rate:0.948864
IR50-TradesArcFace  平均 ssim:0.953079
IR50-TradesArcFace  攻击后 success rate:0.869318
--------------------开始测试 IR50-Softmax-BR---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:54<00:00,  1.63it/s]
IR50-Softmax-BR  benchmark rate:0.996000
IR50-Softmax-BR  攻击前 success rate:0.982620
IR50-Softmax-BR  平均 ssim:0.953029
IR50-Softmax-BR  攻击后 success rate:0.268382
--------------------开始测试 IR50-Softmax-RP---------------------
Load existing checkpoint
100%|██████████| 187/187 [15:22<00:00,  4.93s/it]
IR50-Softmax-RP  benchmark rate:0.994167
IR50-Softmax-RP  攻击前 success rate:0.979947
IR50-Softmax-RP  平均 ssim:0.953052
IR50-Softmax-RP  攻击后 success rate:0.293616
--------------------开始测试 IR50-Softmax-JPEG---------------------
Load existing checkpoint
100%|██████████| 187/187 [02:47<00:00,  1.12it/s]
IR50-Softmax-JPEG  benchmark rate:0.995833
IR50-Softmax-JPEG  攻击前 success rate:0.984291
IR50-Softmax-JPEG  平均 ssim:0.952885
IR50-Softmax-JPEG  攻击后 success rate:0.284759

扰动量4 650轮 集成 ['ArcFace','FaceNet-VGGFace2', 'Mobilenet-stride1','ShuffleNet_V1_GDConv','ResNet50','IR50-CosFace','IR50-ArcFace','IR50-SphereFace']
Load existing checkpoint
100%|██████████| 187/187 [00:55<00:00,  3.36it/s]
FaceNet-casia  benchmark rate:0.981000
FaceNet-casia  攻击前 success rate:0.981785
FaceNet-casia  平均 ssim:0.926402
FaceNet-casia  攻击后 success rate:0.184659
--------------------开始测试 ArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:20<00:00,  2.33it/s]
ArcFace  benchmark rate:0.995000
ArcFace  攻击前 success rate:0.995655
ArcFace  平均 ssim:0.926396
ArcFace  攻击后 success rate:0.071691
--------------------开始测试 SphereFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:43<00:00,  4.32it/s]
SphereFace  benchmark rate:0.981833
SphereFace  攻击前 success rate:0.988803
SphereFace  平均 ssim:0.926183
SphereFace  攻击后 success rate:0.085394
--------------------开始测试 CosFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:44<00:00,  4.18it/s]
CosFace  benchmark rate:0.986833
CosFace  攻击前 success rate:0.984459
CosFace  平均 ssim:0.926352
CosFace  攻击后 success rate:0.118316
--------------------开始测试 MobileFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [00:43<00:00,  4.30it/s]
MobileFace  benchmark rate:0.994500
MobileFace  攻击前 success rate:0.994318
MobileFace  平均 ssim:0.926620
MobileFace  攻击后 success rate:0.038269
--------------------开始测试 IR50-PGDArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:36<00:00,  1.94it/s]
IR50-PGDArcFace  benchmark rate:0.876667
IR50-PGDArcFace  攻击前 success rate:0.810662
IR50-PGDArcFace  平均 ssim:0.926489
IR50-PGDArcFace  攻击后 success rate:0.698362
--------------------开始测试 IR50-TradesArcFace---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:37<00:00,  1.93it/s]
IR50-TradesArcFace  benchmark rate:0.950667
IR50-TradesArcFace  攻击前 success rate:0.951370
IR50-TradesArcFace  平均 ssim:0.926162
IR50-TradesArcFace  攻击后 success rate:0.805314
--------------------开始测试 IR50-Softmax-BR---------------------
Load existing checkpoint
100%|██████████| 187/187 [01:43<00:00,  1.81it/s]
IR50-Softmax-BR  benchmark rate:0.996000
IR50-Softmax-BR  攻击前 success rate:0.984626
IR50-Softmax-BR  平均 ssim:0.926769
IR50-Softmax-BR  攻击后 success rate:0.161096
--------------------开始测试 IR50-Softmax-RP---------------------
Load existing checkpoint
100%|██████████| 187/187 [14:21<00:00,  4.61s/it]
IR50-Softmax-RP  benchmark rate:0.994167
IR50-Softmax-RP  攻击前 success rate:0.982453
IR50-Softmax-RP  平均 ssim:0.926601
IR50-Softmax-RP  攻击后 success rate:0.182821
--------------------开始测试 IR50-Softmax-JPEG---------------------
Load existing checkpoint
100%|██████████| 187/187 [02:19<00:00,  1.34it/s]
IR50-Softmax-JPEG  benchmark rate:0.995833
IR50-Softmax-JPEG  攻击前 success rate:0.983289
IR50-Softmax-JPEG  平均 ssim:0.926193
IR50-Softmax-JPEG  攻击后 success rate:0.168783

lfw:
3 650 ['testir152', 'testmobileface', 'testfacenet']
testir152  benchmark rate:0.999000
testir152  攻击前 success rate:0.989639
testir152  平均 ssim:0.963711
testir152  平均 psnr:36.114491
testir152  平均 mse:15.908763
testir152  攻击后 success rate:0.250501
--------------------开始测试 testmobileface---------------------
100%|██████████| 187/187 [00:38<00:00,  4.80it/s]
testmobileface  benchmark rate:0.999000
testmobileface  攻击前 success rate:0.997828
testmobileface  平均 ssim:0.963729
testmobileface  平均 psnr:36.117445
testmobileface  平均 mse:15.897960
testmobileface  攻击后 success rate:0.153743
--------------------开始测试 testfacenet---------------------
100%|██████████| 187/187 [00:57<00:00,  3.23it/s]
testfacenet  benchmark rate:0.999000
testfacenet  攻击前 success rate:0.991143
testfacenet  平均 ssim:0.963593
testfacenet  平均 psnr:36.113890
testfacenet  平均 mse:15.910955
testfacenet  攻击后 success rate:0.376504
4 650 ['testir152', 'testmobileface', 'testfacenet']
testir152  benchmark rate:0.999000
testir152  攻击前 success rate:0.990976
testir152  平均 ssim:0.942079
testir152  平均 psnr:33.670383
testir152  平均 mse:27.928354
testir152  攻击后 success rate:0.122159
--------------------开始测试 testmobileface---------------------
100%|██████████| 187/187 [00:35<00:00,  5.22it/s]
testmobileface  benchmark rate:0.999000
testmobileface  攻击前 success rate:0.996992
testmobileface  平均 ssim:0.942077
testmobileface  平均 psnr:33.669404
testmobileface  平均 mse:27.934661
testmobileface  攻击后 success rate:0.054646
--------------------开始测试 testfacenet---------------------
100%|██████████| 187/187 [00:52<00:00,  3.53it/s]
testfacenet  benchmark rate:0.999000
testfacenet  攻击前 success rate:0.987634
testfacenet  平均 ssim:0.941981
testfacenet  平均 psnr:33.672212
testfacenet  平均 mse:27.916646
testfacenet  攻击后 success rate:0.208890

celeba-hq:
4 650 无stloss
testir152  benchmark rate:0.990976
testir152  攻击前 success rate:0.985795
testir152  平均 ssim:0.953533
testir152  平均 psnr:33.710672
testir152  平均 mse:27.670735
testir152  攻击后 success rate:0.147727
--------------------开始测试 testmobileface---------------------
100%|██████████| 187/187 [00:42<00:00,  4.44it/s]
testmobileface  benchmark rate:0.996992
testmobileface  攻击前 success rate:0.994652
testmobileface  平均 ssim:0.953395
testmobileface  平均 psnr:33.706063
testmobileface  平均 mse:27.700094
testmobileface  攻击后 success rate:0.096758
--------------------开始测试 testfacenet---------------------
100%|██████████| 187/187 [00:57<00:00,  3.24it/s]
testfacenet  benchmark rate:0.987634
testfacenet  攻击前 success rate:0.990809
testfacenet  平均 ssim:0.953360
testfacenet  平均 psnr:33.709457
testfacenet  平均 mse:27.678418
testfacenet  攻击后 success rate:0.253509

4 650 8白盒一次stloss:
testir152  benchmark rate:0.990976
testir152  攻击前 success rate:0.988636
testir152  平均 ssim:0.965999
testir152  平均 psnr:34.109231
testir152  平均 mse:25.246961
testir152  攻击后 success rate:0.180481
--------------------开始测试 testmobileface---------------------
100%|██████████| 187/187 [00:43<00:00,  4.33it/s]
testmobileface  benchmark rate:0.996992
testmobileface  攻击前 success rate:0.995822
testmobileface  平均 ssim:0.965856
testmobileface  平均 psnr:34.109966
testmobileface  平均 mse:25.244414
testmobileface  攻击后 success rate:0.138536
--------------------开始测试 testfacenet---------------------
100%|██████████| 187/187 [01:49<00:00,  1.70it/s]
testfacenet  benchmark rate:0.987634
testfacenet  攻击前 success rate:0.991310
testfacenet  平均 ssim:0.965874
testfacenet  平均 psnr:34.101645
testfacenet  平均 mse:25.293185
testfacenet  攻击后 success rate:0.292948

ffhq:
4 650 8白盒stloss一次训练
testir152  benchmark rate:0.990976
testir152  攻击前 success rate:0.986631
testir152  平均 ssim:0.965715
testir152  平均 psnr:34.060966
testir152  平均 mse:25.529209
testir152  攻击后 success rate:0.376504
--------------------开始测试 testmobileface---------------------
100%|██████████| 187/187 [00:44<00:00,  4.22it/s]
testmobileface  benchmark rate:0.996992
testmobileface  攻击前 success rate:0.990809
testmobileface  平均 ssim:0.965510
testmobileface  平均 psnr:34.027563
testmobileface  平均 mse:25.744057
testmobileface  攻击后 success rate:0.203710
--------------------开始测试 testfacenet---------------------
100%|██████████| 187/187 [01:28<00:00,  2.12it/s]
testfacenet  benchmark rate:0.987634
testfacenet  攻击前 success rate:0.980615
testfacenet  平均 ssim:0.965486
testfacenet  平均 psnr:34.024551
testfacenet  平均 mse:25.771189
testfacenet  攻击后 success rate:0.371658

4 650 8白盒

'''