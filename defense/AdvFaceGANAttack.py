# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import autograd
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np

from AdvFaceGAN import Generator, ResNetDiscriminator, init_net
from fr_models.get_model import getmodel
from fr_models.config import threshold_lfw
from utils.dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False,
            padding_mode='reflect'
        )
        self.instanceNorm2d = nn.InstanceNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.instanceNorm2d(x)
        if self.activation:
            x = self.activation(x)
        return x


class BasicDeConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, activation=None):
        super().__init__()
        self.deConv = nn.ConvTranspose2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=1,  # 添加 output_padding 以控制输出形状
            bias=False,
        )
        self.instanceNorm2d = nn.InstanceNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        x = self.deConv(x)
        x = self.instanceNorm2d(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       nn.BatchNorm2d(dim, eps=0.001, momentum=0.1, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       nn.BatchNorm2d(dim, eps=0.001, momentum=0.1, affine=True)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = F.relu(out, inplace=True)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



# class Generator1(nn.Module):
#     def __init__(self, is_target=False):
#         super().__init__()
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         k = 64
#         # 修改为输入通道数为 3
#         self.conv1 = BasicConv2d(3, k, 7, stride=1, activation=self.lrelu)  # 修改为 3 通道
#         self.conv2 = BasicConv2d(k, 2 * k, 4, stride=2, activation=self.lrelu)
#         self.conv3 = BasicConv2d(2 * k, 4 * k, 4, stride=2, activation=self.lrelu)
#         self.res_blocks = nn.Sequential(
#             ResnetBlock(4 * k),
#             ResnetBlock(4 * k),
#             ResnetBlock(4 * k),
#             ResnetBlock(4 * k),
#             ResnetBlock(4 * k),
#             ResnetBlock(4 * k),
#         )
#         self.deconv1 = BasicDeConv2d(4 * k, 2 * k, 4, stride=2, activation=self.lrelu)
#         self.deconv2 = BasicDeConv2d(2 * k, k, 4, stride=2, activation=self.lrelu)
#         self.conv_img = nn.ConvTranspose2d(k, 3, 7, stride=1)
#         self.tanh = nn.Tanh()
#
#     def forward(self, sources, targets=None):
#         if targets is not None:
#             net = torch.cat((sources, targets), dim=1)  # 如果需要拼接目标图像
#         else:
#             net = sources
#         # 编码
#         net = self.conv1(net)
#         net = self.conv2(net)
#         net = self.conv3(net)
#         # 残差连接
#         encoded = self.res_blocks(net)
#         # 解码
#         net = self.deconv1(encoded)
#         net = self.deconv2(net)
#         net = self.conv_img(net)
#         perturb = self.tanh(net)
#         # 加性扰动
#         perturb = perturb[:, :, :112, :112]
#
#         output = 2 * torch.clip(perturb + (sources + 1.0) / 2.0, 0, 1) - 1
#         return perturb, output

class Generator1(nn.Module):
    def __init__(self, is_target=False):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        k = 64
        self.conv1 = BasicConv2d(6 if is_target else 3, k, 7, stride=1, activation=self.lrelu)
        self.conv2 = BasicConv2d(k, 2 * k, 4, stride=2, activation=self.lrelu)
        self.conv3 = BasicConv2d(2 * k, 4 * k, 4, stride=2, activation=self.lrelu)
        self.res_blocks = nn.Sequential(
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
        )
        self.deconv1 = BasicDeConv2d(4 * k, 2 * k, 4, stride=2, activation=self.lrelu)
        self.deconv2 = BasicDeConv2d(2 * k, k, 4, stride=2, activation=self.lrelu)
        self.conv_img = nn.ConvTranspose2d(k, 3, 7, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, sources, targets=None):
        if targets is not None:
            net = torch.cat((sources, targets), dim=1)
        else:
            net = sources
        # 编码
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        # 残差连接
        encoded = self.res_blocks(net)
        # 解码
        net = self.deconv1(encoded)
        net = self.deconv2(net)
        net = self.conv_img(net)
        perturb = self.tanh(net)
        # 加性扰动
        perturb = perturb[:, :, :112, :112]
        output = 2 * torch.clip(perturb + (sources + 1.0) / 2.0, 0, 1) - 1
        return perturb, output




# *************************
# AdvFaceGAN main module
# *************************
class AdvFaceGANAttack(nn.Module):
    def __init__(self, config):
        super(AdvFaceGANAttack, self).__init__()
        self.isdebug = False

        self.config = config

        self.mode = config.get('Setting', 'mode')
        print("now mode：" + self.mode)

        # Base Configuration
        gpu = config.getint('Setting', 'gpu')
        self.device = torch.device('cuda:{}'.format(gpu)) if gpu >= 0 else torch.device('cpu')
        self.n_threads = config.getint('Setting', 'n_threads')

        # About Dataset
        self.train_dataset_dir = config.get('Dataset', 'train_dataset_dir')
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # 图片转为Tensor [0,1]
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # 归一化
        ])

        self.test_dataset_dir = config.get('Dataset', 'test_dataset_dir')
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # 图片转为Tensor [0,1]
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # 归一化
        ])

        # About Traing
        self.train_total_epoch = config.getint('Train', 'train_total_epoch')
        self.train_epoch_size = config.getint('Train', 'train_epoch_size')
        self.train_batch_size = config.getint('Train', 'train_batch_size')
        self.train_model_name_list = eval(config.get('Train', 'train_model_name_list'))
        self.lr = config.getfloat('Train', 'lr')

        # pre-trained model setting
        self.models_info = {}
        self.global_step = config.getint('Train', 'global_step')
        self.pretrained_model_dir = config.get('Train', 'pretrained_model_dir')
        self.train_epoch_id = config.getint('Train', 'train_epoch_id')

        # About Validating
        self.val_model_name_list = eval(config.get('Val', 'val_model_name_list'))
        self.val_epoch_size = config.getint('Val', 'val_epoch_size')
        self.val_batch_size = config.getint('Val', 'val_batch_size')

        # About Save
        self.save_dir = config.get('Save', 'save_dir')

        # About Testing
        self.test_model_dir = config.get('Test', 'test_model_dir')
        self.test_epoch_id = config.getint('Test', 'test_epoch_id')
        self.test_model_name_list = eval(config.get('Test', 'test_model_name_list'))
        self.test_batch_size = config.getint('Test', 'test_batch_size')

        # loss storage
        self.d_true_loss = []
        self.d_fake_loss = []
        self.gp_loss = []
        self.D_loss = []

        self.gan_loss = []
        self.adv_loss = []
        self.per_loss = []
        self.st_loss = []
        self.G_loss = []
        # image temp
        self.sources = None
        self.targets = None
        self.perts = None
        self.fakes = None

        # loss factor
        self.per_loss_factor = config.getfloat('Train', 'per_loss_factor')
        self.adv_loss_factor = config.getfloat('Train', 'adv_loss_factor')
        self.st_loss_factor = config.getfloat('Train', 'st_loss_factor')
        self.MAX_PERTURBATION = config.getfloat('Train', 'MAX_PERTURBATION')
        self.MAX_SSIM = config.getfloat('Train', 'MAX_SSIM')

        # Discriminator
        self.discriminator = init_net(ResNetDiscriminator(), self.device)
        # Generator
        self.generator = init_net(Generator(is_target=(self.mode == 'target')), self.device)

        # Discriminator optimizer
        self.discr_opt = torch.optim.Adam(self.discriminator.parameters(),
                                          lr=self.lr,
                                          betas=(0.5, 0.999),
                                          weight_decay=0.0001)
        # Generator optimizer
        self.gen_opt = torch.optim.Adam(self.generator.parameters(),
                                        lr=self.lr,
                                        betas=(0.5, 0.999),
                                        weight_decay=0.0001)

        self.to(self.device)

    def re_init(self):
        # losses
        self.d_true_loss = []
        self.d_fake_loss = []
        self.gp_loss = []
        self.D_loss = []

        self.gan_loss = []
        self.adv_loss = []
        self.adv_loss1 = []
        self.per_loss = []
        self.st_loss = []
        self.G_loss = []
        # images
        self.sources = None
        self.targets = None
        self.perts = None
        self.fakes = None

    def generate_adversarial_image(self, generator, input_tensor, target_tensor=None, device="cuda"):
        """
        使用生成器对输入图像和目标图像张量生成带有扰动的对抗图像。
        generator: 加载的生成器模型
        input_tensor: 输入图像张量，形状为 (B, C, H, W)，通常 B=1
        target_tensor: 目标图像张量（可选），形状为 (B, C, H, W)
        device: 运行生成器的设备（默认 "cuda"）

        返回: 生成的对抗图像张量
        """
        # 确保输入张量和目标张量在正确的设备上
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device) if target_tensor is not None else None

        # 确保生成器也在正确的设备上
        generator.to(device)

        # 使用生成器生成对抗图像
        with torch.no_grad():  # 禁用梯度计算
            # 如果提供了目标图像（target_tensor），则生成器会使用条件生成
            if target_tensor is not None:
                perturb, adversarial_tensor = generator(input_tensor, target_tensor)
            else:
                perturb, adversarial_tensor = generator(input_tensor)

        # 返回生成的对抗图像张量
        return adversarial_tensor

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype=torch.float32, device=self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = Variable(torch.ones((real_samples.shape[0], 1), dtype=torch.float32, device=self.device), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # update Discriminator
    def update_discriminator(self, source_faces, target_faces):
        self.discr_opt.zero_grad()

        source_faces = source_faces.clone().detach().to(self.device)
        #添加扰动
        source_faces1=source_faces
        target_faces = target_faces.clone().detach().to(self.device)

        # 使用生成器生成对抗图像
        model_path ="/media/com/8C922CC9922CBA1A/wyf/advyolo/AdvFaceGAN-main/save/01110_generator.pth"
        generator1 = Generator1(is_target=True).to("cuda")

        # 加载模型时，忽略不匹配的权重
        checkpoint = torch.load(model_path, map_location="cuda")
        generator1.load_state_dict(checkpoint)
        generator1.eval()  # 设置生成器为评估模式
        # 假设 source_faces 是输入图像，确保它有正确的形状
        source_faces = self.generate_adversarial_image(generator1, source_faces,target_faces, device="cuda")

        # Call the generator to generate adversary faces
        _, fake_afters = self.generator(sources=source_faces)

        # Call discriminator to inference embedding
        out_real = self.discriminator(source_faces1)
        loss_true = torch.mean(out_real)
        out_fake = self.discriminator(fake_afters.detach())
        loss_fake = torch.mean(out_fake)

        # Gradient penalty
        lambda_gp = 10
        gradient_penalty = self.compute_gradient_penalty(source_faces1.data, fake_afters.data)
        loss_gp = lambda_gp * gradient_penalty

        # L_D
        loss_d = loss_fake - loss_true + loss_gp

        loss_d.backward()
        self.discr_opt.step()

        # recording
        self.d_true_loss.append(loss_true.item())
        self.d_fake_loss.append(loss_fake.item())
        self.gp_loss.append(loss_gp.item())
        self.D_loss.append(loss_d.item())


    # update Generator
    def update_gen(self, source_faces, target_faces):
        self.gen_opt.zero_grad()

        source_faces = source_faces.clone().detach().to(self.device)
        target_faces1=source_faces
        target_faces = target_faces.clone().detach().to(self.device)

        model_path ="/media/com/8C922CC9922CBA1A/wyf/advyolo/AdvFaceGAN-main/save/01110_generator.pth"
        generator1 = Generator1(is_target=True).to("cuda")

        # 加载模型时，忽略不匹配的权重
        checkpoint = torch.load(model_path, map_location="cuda")
        generator1.load_state_dict(checkpoint)
        generator1.eval()  # 设置生成器为评估模式
        # 假设 source_faces 是输入图像，确保它有正确的形状
        source_faces = self.generate_adversarial_image(generator1, source_faces,target_faces, device="cuda")

        # overall_tar_loss = torch.tensor([], device=self.device)
        # for i, model_name in enumerate(self.train_model_name_list):
        #     if model_name not in self.models_info.keys():
        #         self.models_info[model_name] = getmodel(model_name)
        #     model, img_shape = self.models_info[model_name]
        #     emb_fake_faces = model.forward(
        #         F.interpolate((source_faces * 0.5 + 0.5) * 255, size=img_shape, mode='bilinear'))
        #     emb_target_faces = model.forward(
        #         F.interpolate((target_faces * 0.5 + 0.5) * 255, size=img_shape, mode='bilinear')).detach()
        #     fake_scores = torch.cosine_similarity(emb_fake_faces, emb_target_faces)
        #     tar_loss = fake_scores
        #     overall_tar_loss = torch.cat((overall_tar_loss, tar_loss), dim=0)
        # loss_adv2 = torch.mean(overall_tar_loss)
        # print("loss_adv2",loss_adv2)
        # Call the generator to generate adversary faces
        perturbs, fake_afters = self.generator(sources=source_faces)

        # Call discriminator to inference embedding
        fake_outs = self.discriminator(fake_afters)
        # L_gan
        loss_gan = -torch.mean(fake_outs)

        # L_adv
        overall_tar_loss = torch.tensor([], device=self.device)
        for i, model_name in enumerate(self.train_model_name_list):
            if model_name not in self.models_info.keys():
                self.models_info[model_name] = getmodel(model_name)
            model, img_shape = self.models_info[model_name]
            emb_fake_faces = model.forward(F.interpolate((fake_afters*0.5+0.5)*255, size=img_shape, mode='bilinear'))
            emb_target_faces = model.forward(F.interpolate((target_faces1*0.5+0.5)*255, size=img_shape, mode='bilinear')).detach()
            fake_scores = torch.cosine_similarity(emb_fake_faces, emb_target_faces)
            tar_loss = 1 - fake_scores
            overall_tar_loss = torch.cat((overall_tar_loss, tar_loss), dim=0)
        loss_adv = self.adv_loss_factor * (torch.mean(overall_tar_loss))

        overall_tar_loss = torch.tensor([], device=self.device)
        for i, model_name in enumerate(self.train_model_name_list):
            if model_name not in self.models_info.keys():
                self.models_info[model_name] = getmodel(model_name)
            model, img_shape = self.models_info[model_name]
            emb_fake_faces = model.forward(
                F.interpolate((fake_afters * 0.5 + 0.5) * 255, size=img_shape, mode='bilinear'))
            emb_target_faces = model.forward(
                F.interpolate((target_faces * 0.5 + 0.5) * 255, size=img_shape, mode='bilinear')).detach()
            fake_scores = torch.cosine_similarity(emb_fake_faces, emb_target_faces)
            tar_loss = fake_scores
            overall_tar_loss = torch.cat((overall_tar_loss, tar_loss), dim=0)
        loss_adv1 = self.adv_loss_factor * (torch.mean(overall_tar_loss))
        # L_pert
        loss_per = self.per_loss_factor * torch.mean(
            torch.maximum(torch.zeros(perturbs.shape[0], device=self.device) + self.MAX_PERTURBATION,
                          torch.norm(perturbs.view(perturbs.shape[0], -1), dim=1)))

        # L_st
        loss_st = self.st_loss_factor * torch.mean(
            torch.maximum(torch.zeros(perturbs.shape[0], device=self.device) + (1-self.MAX_SSIM),
                          1-ssim(target_faces1, fake_afters)))

        # L_G
        loss_g = loss_gan + loss_adv + loss_per + loss_st+loss_adv1

        loss_g.backward()
        self.gen_opt.step()

        # recording
        self.gan_loss.append(loss_gan.item())
        self.adv_loss.append(loss_adv.item())
        self.adv_loss1.append(loss_adv1.item())
        self.per_loss.append(loss_per.item())
        self.st_loss.append(loss_st.item())
        self.G_loss.append(loss_g.item())

        self.sources = source_faces
        self.targets = target_faces
        self.perts = perturbs
        self.fakes = fake_afters

    @torch.no_grad()
    def visualization(self, now_epoch, now_batch, batch_num):
        print(" EPOCH[%d] Batch[%d]/[%d]:"
              "D_loss: %.2f (True_loss: %.2f, Fake_loss: %.2f, GP_loss:%.2f) "
              "G_loss: %.2f (GAN_loss: %.2f, Adv_loss: %.2f,Adv_loss1: %.2f, Per_loss: %.2f, St_loss: %.2f)" %
              (
                  now_epoch, now_batch, batch_num,
                  self.D_loss[-1],
                  self.d_true_loss[-1],
                  self.d_fake_loss[-1],
                  self.gp_loss[-1],
                  self.G_loss[-1],
                  self.gan_loss[-1],
                  self.adv_loss[-1],
                  self.adv_loss1[-1],
                  self.per_loss[-1],
                  self.st_loss[-1],
              ))
        if now_batch == batch_num:
            # Create the log summary writer
            writer = SummaryWriter(self.save_dir+'/log')
            # Recording scalar data
            writer.add_scalar("loss/D_loss", np.mean(self.D_loss), self.global_step)
            writer.add_scalar("loss/d_true_loss", np.mean(self.d_true_loss), self.global_step)
            writer.add_scalar("loss/d_fake_loss", np.mean(self.d_fake_loss), self.global_step)
            writer.add_scalar("loss/gp_loss", np.mean(self.gp_loss), self.global_step)

            writer.add_scalar("loss/G_loss", np.mean(self.G_loss), self.global_step)
            writer.add_scalar("loss/gan_loss", np.mean(self.gan_loss), self.global_step)
            writer.add_scalar("loss/adv_loss", np.mean(self.adv_loss), self.global_step)
            writer.add_scalar("loss/adv_loss1", np.mean(self.adv_loss1), self.global_step)
            writer.add_scalar("loss/per_loss", np.mean(self.per_loss), self.global_step)
            writer.add_scalar("loss/st_loss", np.mean(self.st_loss), self.global_step)
            # Recording image data
            # writer.add_image("image/source{0}".format(self.global_step), self.sources[0], self.global_step)
            # writer.add_image("image/target{0}".format(self.global_step), self.targets[0], self.global_step)
            # writer.add_image("image/pert{0}".format(self.global_step), self.perts[0], self.global_step)
            # writer.add_image("image/fake{0}".format(self.global_step), self.fakes[0], self.global_step)
            writer.flush()
            writer.close()

            if self.isdebug:
                os.makedirs("./log", exist_ok=True)
                torchvision.utils.save_image(self.sources[0:4]*0.5+0.5, './log/source.png', nrow=1)
                torchvision.utils.save_image(self.targets[0:4]*0.5+0.5, './log/target.png', nrow=1)
                torchvision.utils.save_image(self.perts[0:4]*0.5+0.5, './log/pert.png', nrow=1)
                torchvision.utils.save_image(self.fakes[0:4]*0.5+0.5, './log/fake.png', nrow=1)

            self.global_step += 1

    def train_epoch(self, epoch_id, train_set):
        # reset before you start new training
        self.re_init()
        print("----------------------epoch {0}：start training-----------------------".format(epoch_id))
        for batch_id in range(0, self.train_epoch_size):
            batch = train_set.pop_batch_queue()
            # get attacker face and victim face
            source_faces = batch['sources']
            target_faces = batch['targets']
            # update Discriminator
            self.update_discriminator(source_faces, target_faces)
            # update Generator
            self.update_gen(source_faces, target_faces)
            # Save faces and output training logs
            self.visualization(epoch_id, batch_id + 1, self.train_epoch_size)

    @torch.no_grad()
    def validate_epoch(self, epoch_id, validate_set):
        print("----------------------epoch {0}：start validating-----------------------".format(epoch_id))
        for model_name in self.val_model_name_list:
            print("--------------------epoch {0}：start validating {1}---------------------".format(epoch_id, model_name))
            ssim_scores = []
            psnr_socres = []
            mse_scores = []
            src_simi_scores = []
            rob_simi_scores = []
            if model_name not in self.models_info.keys():
                self.models_info[model_name] = getmodel(model_name)
            model, img_shape = self.models_info[model_name]
            for _ in tqdm(range(0, self.val_epoch_size), total=self.val_epoch_size):
                # get batch data
                batch = validate_set.pop_batch_queue()
                # call the generator to generate adversary faces
                source_faces = batch['sources'].to(self.device)
                target_faces = batch['targets'].to(self.device)
                _, fake_afters = self.generator(sources=source_faces, targets=None if self.mode == "untarget" else target_faces)
                # computed Visual Quality Metric
                source_faces = (source_faces * 0.5 + 0.5) * 255
                fake_afters = (fake_afters * 0.5 + 0.5) * 255
                target_faces = (target_faces * 0.5 + 0.5) * 255
                ssim_scores.append(ssim(source_faces, fake_afters, data_range=255.0).item())
                psnr_socres.append(psnr(source_faces, fake_afters).item())
                mse_scores.append(F.mse_loss(source_faces, fake_afters).item())
                # extract face embedding
                emb_source = model.forward(F.interpolate(source_faces, size=img_shape, mode='bilinear'))
                emb_fake_after = model.forward(F.interpolate(fake_afters, size=img_shape, mode='bilinear'))
                emb_target = model.forward(F.interpolate(target_faces, size=img_shape, mode='bilinear'))
                # computed Cosine similarity
                src_simi_scores.extend(torch.cosine_similarity(emb_source, emb_target).tolist())
                rob_simi_scores.extend(torch.cosine_similarity(emb_fake_after, emb_target).tolist())

            th = threshold_lfw[model_name]['cos']
            src_success_rate = sum(
                score > th if self.mode == "untarget" else score < th for score in src_simi_scores) / len(
                src_simi_scores)
            rob_success_rate = sum(
                score > th if self.mode == "untarget" else score < th for score in rob_simi_scores) / len(
                rob_simi_scores)
            print(model_name, " benchmark rate:%f" % threshold_lfw[model_name]['cos_acc'])
            print(model_name, " before attack success rate:%f" % src_success_rate)
            print(model_name, " average ssim:%f" % np.mean(ssim_scores))
            print(model_name, " average psnr:%f" % np.mean(psnr_socres))
            print(model_name, " average mse:%f" % np.mean(mse_scores))
            print(model_name, " after attack success rate:%f" % rob_success_rate)

    @torch.no_grad()
    def save_model(self, epoch_id):
        if epoch_id % 10 == 0:  # Add conditions for use less storage space
            print("----------------------epoch{0}：saving model-----------------------".format(epoch_id))
            # save model
            os.makedirs(self.save_dir+'/model', exist_ok=True)
            # Save the generator's parameters
            torch.save(self.generator.state_dict(), '%s/%05d_generator.pth' % (self.save_dir+'/model', epoch_id))
            # Save the discriminator's parameters
            torch.save(self.discriminator.state_dict(), '%s/%05d_discriminator.pth' % (self.save_dir+'/model', epoch_id))

    def start_training(self):
        print("---------------------cut train set and test set-----------------")
        train_set, validate_set = Dataset(self.train_dataset_dir, self.mode).separate_by_ratio(0.9)
        print("-----------------------start batch queue------------------------")
        train_set.start_batch_queue(
            self.train_batch_size,
            batch_format="random_samples",
            transforms=self.train_transforms,
            num_threads=self.n_threads,
        )
        if self.isdebug:
            validate_set.start_batch_queue(
                self.val_batch_size,
                batch_format="random_samples",
                transforms=self.train_transforms,
                num_threads=self.n_threads
            )
        if self.global_step != 0:
            print("--------------------Load the pre-trained model---------------------")
            self.load_model(self.pretrained_model_dir, self.train_epoch_id)
        print("---------------------------------------------------------")
        print("--------------------Start model training-----------------")
        print("---------------------------------------------------------")
        print("--------- ----Save training configuration----------------")
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_dir + '/train_config.ini', 'w', encoding='utf-8') as fp:
            self.config.write(fp)
        for epoch_id in range(0 if self.global_step == 0 else self.train_epoch_id + 1, self.train_total_epoch):
            # 训练
            self.train_epoch(epoch_id, train_set)
            if self.isdebug:
                # 验证
                self.validate_epoch(epoch_id, validate_set)
            # 保存
            self.save_model(epoch_id)
        print("---------------------------------------------------------")
        print("------------------End of model training------------------")
        print("---------------------------------------------------------")

    @torch.no_grad()
    def start_testing(self):
        print("---------------------Load test set-----------------------")
        test_set = Dataset(self.test_dataset_dir, self.mode)
        print("------------------start batch queue----------------------")
        test_set.start_batch_queue(
            self.test_batch_size,
            batch_format="random_samples",
            transforms=self.test_transforms,
            num_threads=self.n_threads,
        )
        # Load test parameters
        self.load_model(self.test_model_dir, self.test_epoch_id)
        print("---------------------------------------------------------")
        print("------------------Start model evaluating--------------------")
        print("---------------------------------------------------------")
        test_epoch_size = 6000 // self.test_batch_size
        # Calculate each model's metric
        for model_name in self.test_model_name_list:
            print("-----------------start evaluate {0}-------------------".format(model_name))
            ssim_scores = []
            psnr_socres = []
            mse_scores = []
            src_simi_scores = []
            rob_simi_scores = []
            if model_name not in self.models_info.keys():
                self.models_info[model_name] = getmodel(model_name)
            model, img_shape = self.models_info[model_name]
            for _ in tqdm(range(test_epoch_size)):
                # get batch data
                batch = test_set.pop_batch_queue()
                # Call the generator to generate adversary faces
                source_faces = batch['sources'].to(self.device)
                target_faces = batch['targets'].to(self.device)
                perts, fake_afters = self.generator(sources=source_faces, targets=None if self.mode == "untarget" else target_faces)
                # calculate visual quality metric
                source_faces = (source_faces*0.5+0.5)*255
                fake_afters = (fake_afters*0.5+0.5)*255
                target_faces = (target_faces*0.5+0.5)*255
                ssim_scores.append(ssim(source_faces, fake_afters, data_range=255.0).item())
                psnr_socres.append(psnr(source_faces, fake_afters).item())
                mse_scores.append(F.mse_loss(source_faces, fake_afters).item())
                # extract face embedding
                emb_source = model.forward(F.interpolate(source_faces, size=img_shape, mode='bilinear'))
                emb_fake_after = model.forward(F.interpolate(fake_afters, size=img_shape, mode='bilinear'))
                emb_target = model.forward(F.interpolate(target_faces, size=img_shape, mode='bilinear'))
                #  evaluation cosine similarity
                src_simi_scores.extend(torch.cosine_similarity(emb_source, emb_target).tolist())
                rob_simi_scores.extend(torch.cosine_similarity(emb_fake_after, emb_target).tolist())
            th = threshold_lfw[model_name]['cos']
            src_success_rate = sum(
                score > th if self.mode == "untarget" else score < th for score in src_simi_scores) / len(
                src_simi_scores)
            rob_success_rate = sum(
                score > th if self.mode == "untarget" else score < th for score in rob_simi_scores) / len(
                rob_simi_scores)
            print(model_name, " benchmark rate:%f" % threshold_lfw[model_name]['cos_acc'])
            print(model_name, " before attack success rate:%f" % src_success_rate)
            print(model_name, " average ssim:%f" % np.mean(ssim_scores))
            print(model_name, " average psnr:%f" % np.mean(psnr_socres))
            print(model_name, " average mse:%f" % np.mean(mse_scores))
            print(model_name, " after attack success rate:%f" % rob_success_rate)
            print(model_name, " attack success rate:%f" % (src_success_rate-rob_success_rate))
        print("---------------------------------------------------------")
        print("-----------------End of model evaluate-------------------")
        print("---------------------------------------------------------")

    @torch.no_grad()
    def generate_fake(self, source_img_path, target_img_path=None):
        # Load the test model parameters
        self.load_model(self.test_model_dir, self.test_epoch_id)
        source_face = torch.unsqueeze(self.test_transforms(Image.open(source_img_path).convert('RGB')), dim=0).to(self.device)
        if self.mode == "target":
            target_face = torch.unsqueeze(self.test_transforms(Image.open(target_img_path).convert('RGB')), dim=0).to(self.device)
        time1 = time.time()
        perts, fake_after = self.generator.forward(sources=source_face, targets=None if self.mode == "untarget" else target_face)
        time2 = time.time()
        print("time consume: ", time2-time1)
        print("ssim: ", ssim(source_face * 0.5 + 0.5, fake_after * 0.5 + 0.5, data_range=1.0).item())
        print("mse: ", F.mse_loss((source_face*0.5+0.5)*255, (fake_after*0.5+0.5)*255).item())
        os.makedirs("./test", exist_ok=True)
        torchvision.utils.save_image(source_face[0:1]*0.5+0.5, './test/source.png', nrow=1)
        torchvision.utils.save_image(target_face[0:1]*0.5+0.5, './test/target.png', nrow=1)
        torchvision.utils.save_image(perts[0:1]*0.5+0.5, './test/pert.png', nrow=1)
        torchvision.utils.save_image(fake_after[0:1]*0.5+0.5, './test/fake.png', nrow=1)
        return perts, fake_after

    def load_model(self, model_dir, epoch_id):
        # Loads the pretrained model parameters specified in the configuration file
        model_generator_dict = torch.load(model_dir + '/' + '%05d_generator.pth' % epoch_id, weights_only=True)
        model_discriminator_dict = torch.load(model_dir + '/' + '%05d_discriminator.pth' % epoch_id, weights_only=True)
        self.generator.load_state_dict(model_generator_dict)
        self.discriminator.load_state_dict(model_discriminator_dict)