import os
import sys

import torch

from fr_models.MobileFace import MobileFace
from fr_models.Mobilenet import Mobilenet
from fr_models.MobilenetV2 import MobilenetV2
from fr_models.ResNet import resnet
from fr_models.ShuffleNetV2 import ShuffleNetV2
from fr_models.ShuffleNet import ShuffleNetV1
from fr_models.CosFace import CosFace
from fr_models.SphereFace import SphereFace
from fr_models.FaceNet import FaceNet
from fr_models.ArcFace import ArcFace
from fr_models.IR import IR

import fr_models.test.irse as testirse
import fr_models.test.ir152 as testir152
import fr_models.test.facenet as testfacenet


def getmodel(face_model, **kwargs):
    """
        select the face model according to its name
        :param face_model: string
        return:
        a model class
    """
    img_shape = (112, 112)
    if face_model == 'MobileFace':
        model = MobileFace(**kwargs)
    elif face_model == 'Mobilenet':
        model = Mobilenet(**kwargs)
    elif face_model == 'Mobilenet-stride1':
        model = Mobilenet(stride=1, **kwargs)
    elif face_model == 'MobilenetV2':
        model = MobilenetV2(**kwargs)
    elif face_model == 'MobilenetV2-stride1':
        model = MobilenetV2(stride=1, **kwargs)
    elif face_model == 'ResNet50':
        model = resnet(depth=50, **kwargs)
    elif face_model == 'ResNet50-casia':
        model = resnet(depth=50, dataset='casia', **kwargs)
    elif face_model == 'ShuffleNet_V1_GDConv':
        model = ShuffleNetV1(pooling='GDConv', **kwargs)
    elif face_model == 'ShuffleNet_V2_GDConv-stride1':
        model = ShuffleNetV2(stride=1, pooling='GDConv', **kwargs)
    elif face_model == 'CosFace':
        model = CosFace(**kwargs)
        img_shape = (112, 96)
    elif face_model == 'SphereFace':
        model = SphereFace(**kwargs)
        img_shape = (112, 96)
    elif face_model == 'FaceNet-VGGFace2':
        model = FaceNet(dataset='vggface2', use_prewhiten=False, **kwargs)
        img_shape = (160, 160)
    elif face_model == 'FaceNet-casia':
        model = FaceNet(dataset='casia-webface', use_prewhiten=False, **kwargs)
        img_shape = (160, 160)
    elif face_model == 'ArcFace':
        model = ArcFace(**kwargs)
    elif face_model == 'IR50-Softmax':
        model = IR(loss='Softmax', **kwargs)
    elif face_model == 'IR50-Softmax-BR':
        model = IR(loss='Softmax', transform='BitReudction', **kwargs)
    elif face_model == 'IR50-Softmax-RP':
        model = IR(loss='Softmax', transform='Randomization', **kwargs)
    elif face_model == 'IR50-Softmax-JPEG':
        model = IR(loss='Softmax', transform='JPEG', **kwargs)
    elif face_model == 'IR50-PGDSoftmax':
        model = IR(loss='PGDSoftmax', **kwargs)
    elif face_model == 'IR50-TradesSoftmax':
        model = IR(loss='TradesSoftmax', **kwargs)
    elif face_model == 'IR50-CosFace':
        model = IR(loss='CosFace', **kwargs)
    elif face_model == 'IR50-TradesCosFace':
        model = IR(loss='TradesCosFace', **kwargs)
    elif face_model == 'IR50-PGDCosFace':
        model = IR(loss='PGDCosFace', **kwargs)
    elif face_model == 'IR50-Am':
        model = IR(loss='Am', **kwargs)
    elif face_model == 'IR50-PGDAm':
        model = IR(loss='PGDAm', **kwargs)
    elif face_model == 'IR50-ArcFace':
        model = IR(loss='ArcFace', **kwargs)
    elif face_model == 'IR50-PGDArcFace':
        model = IR(loss='PGDArcFace', **kwargs)
    elif face_model == 'IR50-TradesArcFace':
        model = IR(loss='TradesArcFace', **kwargs)
    elif face_model == 'IR50-SphereFace':
        model = IR(loss='SphereFace', **kwargs)
    elif face_model == 'IR50-PGDSphereFace':
        model = IR(loss='PGDSphereFace', **kwargs)
    elif face_model == 'CASIA-Softmax':
        model = IR(loss='CASIA-Softmax', **kwargs)
    elif face_model == 'CASIA-CosFace':
        model = IR(loss='CASIA-CosFace', **kwargs)
    elif face_model == 'CASIA-ArcFace':
        model = IR(loss='CASIA-ArcFace', **kwargs)
    elif face_model == 'CASIA-SphereFace':
        model = IR(loss='CASIA-SphereFace', **kwargs)
    elif face_model == 'CASIA-Am':
        model = IR(loss='CASIA-Am', **kwargs)
    # 为了保证测试的可靠性，与adv-attribute对比时使用他所用的模型文件
    elif face_model == 'testfacenet':
        img_shape = (160, 160)
        model = testfacenet.InceptionResnetV1(num_classes=8631, device='cuda:0')
        model.load_state_dict(torch.load('./fr_models/ckpts/facenet.pth', weights_only=True))
        model.eval()
        model.to("cuda:0")
    elif face_model == 'testir152':
        model = testir152.IR_152((112, 112))
        model.load_state_dict(torch.load('./fr_models/ckpts/ir152.pth', weights_only=True))
        model.eval()
        model.to("cuda:0")
    elif face_model == 'testirse50':
        model = testirse.Backbone(50, 0.6, 'ir_se')
        model.load_state_dict(torch.load('./fr_models/ckpts/irse50.pth', weights_only=True))
        model.eval()
        model.to("cuda:0")
    elif face_model == "testmobileface":
        model = testirse.MobileFaceNet(512)
        model.load_state_dict(torch.load('./fr_models/ckpts/mobile_face.pth', weights_only=True))
        model.eval()
        model.to("cuda:0")
    else:
        print(face_model)
        raise NotImplementedError
    return model, img_shape
