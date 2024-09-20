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


# *************************
# AdvFaceGAN 主模型
# *************************
class AdvFaceGANAttack(nn.Module):
    def __init__(self, config):
        super(AdvFaceGANAttack, self).__init__()
        self.isdebug = False

        self.config = config

        self.mode = config.get('Setting', 'mode')
        print("当前模式：" + self.mode)

        gpu = config.getint('Setting', 'gpu')
        self.device = torch.device('cuda:{}'.format(gpu)) if gpu >= 0 else torch.device('cpu')
        self.n_threads = config.getint('Setting', 'n_threads')

        # 数据集相关
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

        # 模型训练相关
        self.train_total_epoch = config.getint('Train', 'train_total_epoch')
        self.train_epoch_size = config.getint('Train', 'train_epoch_size')
        self.train_batch_size = config.getint('Train', 'train_batch_size')
        self.train_model_name_list = eval(config.get('Train', 'train_model_name_list'))
        self.lr = config.getfloat('Train', 'lr')

        # 预训练模型
        self.global_step = config.getint('Train', 'global_step')
        self.pretrained_model_dir = config.get('Train', 'pretrained_model_dir')
        self.train_epoch_id = config.getint('Train', 'train_epoch_id')

        # 模型验证相关
        self.val_model_name_list = eval(config.get('Val', 'val_model_name_list'))
        self.val_epoch_size = config.getint('Val', 'val_epoch_size')
        self.val_batch_size = config.getint('Val', 'val_batch_size')

        # FR预训练模型
        self.models_info = {}

        # 模型存储相关
        self.save_dir = config.get('Save', 'save_dir')

        # 模型测试相关
        self.test_model_dir = config.get('Test', 'test_model_dir')
        self.test_epoch_id = config.getint('Test', 'test_epoch_id')
        self.test_model_name_list = eval(config.get('Test', 'test_model_name_list'))
        self.test_batch_size = config.getint('Test', 'test_batch_size')

        # 损失计算
        self.d_true_loss = []
        self.d_fake_loss = []
        self.gp_loss = []
        self.D_loss = []

        self.gan_loss = []
        self.adv_loss = []
        self.per_loss = []
        self.st_loss = []
        self.G_loss = []
        # 结果图
        self.sources = None
        self.targets = None
        self.perts = None
        self.fakes = None

        # 损失函数相关
        self.per_loss_factor = config.getfloat('Train', 'per_loss_factor')
        self.adv_loss_factor = config.getfloat('Train', 'adv_loss_factor')
        self.st_loss_factor = config.getfloat('Train', 'st_loss_factor')
        self.MAX_PERTURBATION = config.getfloat('Train', 'MAX_PERTURBATION')
        self.MAX_SSIM = config.getfloat('Train', 'MAX_SSIM')

        # 判别器
        self.discriminator = init_net(ResNetDiscriminator(), self.device)
        # 生成器
        self.generator = init_net(Generator(is_target=(self.mode == 'target')), self.device)

        # 判别器优化器
        self.discr_opt = torch.optim.Adam(self.discriminator.parameters(),
                                          lr=self.lr,
                                          betas=(0.5, 0.999),
                                          weight_decay=0.0001)
        # 生成器优化器
        self.gen_opt = torch.optim.Adam(self.generator.parameters(),
                                        lr=self.lr,
                                        betas=(0.5, 0.999),
                                        weight_decay=0.0001)

        self.to(self.device)

    def re_init(self):
        # 损失
        self.d_true_loss = []
        self.d_fake_loss = []
        self.gp_loss = []
        self.D_loss = []

        self.gan_loss = []
        self.adv_loss = []
        self.per_loss = []
        self.st_loss = []
        self.G_loss = []
        # 结果图
        self.sources = None
        self.targets = None
        self.perts = None
        self.fakes = None

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

    # 更新判别器
    def update_discriminator(self, source_faces, target_faces):
        self.discr_opt.zero_grad()

        source_faces = source_faces.clone().detach().to(self.device)
        target_faces = target_faces.clone().detach().to(self.device)
        # 调用生成器生成对抗人脸
        _, fake_afters = self.generator(sources=source_faces,targets=None if self.mode == "untarget" else target_faces)

        # 调用判别器推理
        out_real = self.discriminator(source_faces)
        loss_true = torch.mean(out_real)
        out_fake = self.discriminator(fake_afters.detach())
        loss_fake = torch.mean(out_fake)

        # 梯度惩罚
        lambda_gp = 10
        gradient_penalty = self.compute_gradient_penalty(source_faces.data, fake_afters.data)
        loss_gp = lambda_gp * gradient_penalty

        # 判别损失累加
        loss_d = loss_fake - loss_true + loss_gp

        loss_d.backward()
        self.discr_opt.step()

        # 记录信息
        self.d_true_loss.append(loss_true.item())
        self.d_fake_loss.append(loss_fake.item())
        self.gp_loss.append(loss_gp.item())
        self.D_loss.append(loss_d.item())


    # 更新生成器
    def update_gen(self, source_faces, target_faces):
        self.gen_opt.zero_grad()

        source_faces = source_faces.clone().detach().to(self.device)
        target_faces = target_faces.clone().detach().to(self.device)

        # 调用生成器生成对抗人脸
        perturbs, fake_afters = self.generator(sources=source_faces, targets=None if self.mode == "untarget" else target_faces)

        # 调用判别器推理
        fake_outs = self.discriminator(fake_afters)
        # 判别损失
        loss_gan = -torch.mean(fake_outs)

        # 对抗损失 集成模型
        overall_tar_loss = torch.tensor([], device=self.device)
        for i, model_name in enumerate(self.train_model_name_list):
            if model_name not in self.models_info.keys():
                self.models_info[model_name] = getmodel(model_name)
            model, img_shape = self.models_info[model_name]
            emb_fake_faces = model.forward(F.interpolate((fake_afters*0.5+0.5)*255, size=img_shape, mode='bilinear'))
            emb_target_faces = model.forward(F.interpolate((target_faces*0.5+0.5)*255, size=img_shape, mode='bilinear')).detach()
            fake_scores = torch.cosine_similarity(emb_fake_faces, emb_target_faces)
            tar_loss = fake_scores if self.mode == "untarget" else 1 - fake_scores
            overall_tar_loss = torch.cat((overall_tar_loss, tar_loss), dim=0)
        loss_adv = self.adv_loss_factor * torch.mean(overall_tar_loss)

        # 扰动损失 二范数 控制幅度 尽可能小，但不低于指定量以激活
        loss_per = self.per_loss_factor * torch.mean(
            torch.maximum(torch.zeros(perturbs.shape[0], device=self.device) + self.MAX_PERTURBATION,
                          torch.norm(perturbs.view(perturbs.shape[0], -1), dim=1)))

        # 结构损失  1-ssim尽可能小，
        loss_st = self.st_loss_factor * torch.mean(
            torch.maximum(torch.zeros(perturbs.shape[0], device=self.device) + (1-self.MAX_SSIM),
                          1-ssim(source_faces, fake_afters)))

        # 生成损失累加
        loss_g = loss_gan + loss_adv + loss_per + loss_st

        loss_g.backward()
        self.gen_opt.step()

        # 记录
        self.gan_loss.append(loss_gan.item())
        self.adv_loss.append(loss_adv.item())
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
              "G_loss: %.2f (GAN_loss: %.2f, Adv_loss: %.2f, Per_loss: %.2f, St_loss: %.2f)" %
              (
                  now_epoch, now_batch, batch_num,
                  self.D_loss[-1],
                  self.d_true_loss[-1],
                  self.d_fake_loss[-1],
                  self.gp_loss[-1],
                  self.G_loss[-1],
                  self.gan_loss[-1],
                  self.adv_loss[-1],
                  self.per_loss[-1],
                  self.st_loss[-1],
              ))
        if now_batch == batch_num:
            # 创建日志摘要写入器
            writer = SummaryWriter(self.save_dir+'/log')
            # 记录标量数据
            writer.add_scalar("loss/D_loss", np.mean(self.D_loss), self.global_step)
            writer.add_scalar("loss/d_true_loss", np.mean(self.d_true_loss), self.global_step)
            writer.add_scalar("loss/d_fake_loss", np.mean(self.d_fake_loss), self.global_step)
            writer.add_scalar("loss/gp_loss", np.mean(self.gp_loss), self.global_step)

            writer.add_scalar("loss/G_loss", np.mean(self.G_loss), self.global_step)
            writer.add_scalar("loss/gan_loss", np.mean(self.gan_loss), self.global_step)
            writer.add_scalar("loss/adv_loss", np.mean(self.adv_loss), self.global_step)
            writer.add_scalar("loss/per_loss", np.mean(self.per_loss), self.global_step)
            writer.add_scalar("loss/st_loss", np.mean(self.st_loss), self.global_step)
            # 记录图像数据
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
        # 开始训练前清空旧损失
        self.re_init()
        print("----------------------第{0}轮：开始训练-----------------------".format(epoch_id))
        for batch_id in range(0, self.train_epoch_size):
            batch = train_set.pop_batch_queue()
            # 获取对应的批次数据
            source_faces = batch['sources']
            target_faces = batch['targets']
            # 更新判别器
            self.update_discriminator(source_faces, target_faces)
            # 更新生成器
            self.update_gen(source_faces, target_faces)
            # 保存可视化图像并输出训练日志
            self.visualization(epoch_id, batch_id + 1, self.train_epoch_size)

    @torch.no_grad()
    def validate_epoch(self, epoch_id, validate_set):
        print("----------------------第{0}轮：开始验证-----------------------".format(epoch_id))
        for model_name in self.val_model_name_list:
            print("--------------------第{0}轮：开始验证 {1}---------------------".format(epoch_id, model_name))
            ssim_scores = []
            psnr_socres = []
            mse_scores = []
            src_simi_scores = []
            rob_simi_scores = []
            if model_name not in self.models_info.keys():
                self.models_info[model_name] = getmodel(model_name)
            model, img_shape = self.models_info[model_name]
            for _ in tqdm(range(0, self.val_epoch_size), total=self.val_epoch_size):
                # 获取对应的批次数据
                batch = validate_set.pop_batch_queue()
                # 调用生成器生成对抗人脸
                source_faces = batch['sources'].to(self.device)
                target_faces = batch['targets'].to(self.device)
                _, fake_afters = self.generator(sources=source_faces, targets=None if self.mode == "untarget" else target_faces)
                # 计算视觉指标
                source_faces = (source_faces * 0.5 + 0.5) * 255
                fake_afters = (fake_afters * 0.5 + 0.5) * 255
                target_faces = (target_faces * 0.5 + 0.5) * 255
                ssim_scores.append(ssim(source_faces, fake_afters, data_range=255.0).item())
                psnr_socres.append(psnr(source_faces, fake_afters).item())
                mse_scores.append(F.mse_loss(source_faces, fake_afters).item())
                # 提取人脸嵌入
                emb_source = model.forward(F.interpolate(source_faces, size=img_shape, mode='bilinear'))
                emb_fake_after = model.forward(F.interpolate(fake_afters, size=img_shape, mode='bilinear'))
                emb_target = model.forward(F.interpolate(target_faces, size=img_shape, mode='bilinear'))
                # 余弦相似度评估
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
            print(model_name, " 攻击前 success rate:%f" % src_success_rate)
            print(model_name, " 平均 ssim:%f" % np.mean(ssim_scores))
            print(model_name, " 平均 psnr:%f" % np.mean(psnr_socres))
            print(model_name, " 平均 mse:%f" % np.mean(mse_scores))
            print(model_name, " 攻击后 success rate:%f" % rob_success_rate)

    @torch.no_grad()
    def save_model(self, epoch_id):
        if epoch_id % 10 == 0:  # 加条件少存一点
            print("----------------------第{0}轮：保存模型-----------------------".format(epoch_id))
            # 保存模型
            os.makedirs(self.save_dir+'/model', exist_ok=True)
            # 保存生成器的参数
            torch.save(self.generator.state_dict(), '%s/%05d_generator.pth' % (self.save_dir+'/model', epoch_id))
            # 保存判别器的参数
            torch.save(self.discriminator.state_dict(), '%s/%05d_discriminator.pth' % (self.save_dir+'/model', epoch_id))

    def start_training(self):
        print("---------------------划分训练集与验证集----------------------")
        train_set, validate_set = Dataset(self.train_dataset_dir, self.mode).separate_by_ratio(0.9)
        print("-----------------------启动分批队列------------------------")
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
            print("-----------------------加载预训练模型------------------------")
            self.load_model(self.pretrained_model_dir, self.train_epoch_id)
        print("---------------------------------------------------------")
        print("-----------------------开始模型训练------------------------")
        print("---------------------------------------------------------")
        print("-----------------------保存训练配置------------------------")
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
        print("-----------------------模型训练结束------------------------")
        print("---------------------------------------------------------")

    @torch.no_grad()
    def start_testing(self):
        print("------------------------加载测试集-------------------------")
        test_set = Dataset(self.test_dataset_dir, self.mode)
        print("-----------------------启动分批队列------------------------")
        test_set.start_batch_queue(
            self.test_batch_size,
            batch_format="random_samples",
            transforms=self.test_transforms,
            num_threads=self.n_threads,
        )
        # 加载测试模型参数
        self.load_model(self.test_model_dir, self.test_epoch_id)
        print("---------------------------------------------------------")
        print("-----------------------开始模型测试------------------------")
        print("---------------------------------------------------------")
        test_epoch_size = 6000 // self.test_batch_size
        # 各模型相似分数字典
        for model_name in self.test_model_name_list:
            print("--------------------开始测试 {0}---------------------".format(model_name))
            ssim_scores = []
            psnr_socres = []
            mse_scores = []
            src_simi_scores = []
            rob_simi_scores = []
            if model_name not in self.models_info.keys():
                self.models_info[model_name] = getmodel(model_name)
            model, img_shape = self.models_info[model_name]
            for _ in tqdm(range(test_epoch_size)):
                # 获取对应的批次数据
                batch = test_set.pop_batch_queue()
                # 调用生成器生成对抗人脸
                source_faces = batch['sources'].to(self.device)
                target_faces = batch['targets'].to(self.device)
                perts, fake_afters = self.generator(sources=source_faces, targets=None if self.mode == "untarget" else target_faces)
                # 计算视觉指标
                source_faces = (source_faces*0.5+0.5)*255
                fake_afters = (fake_afters*0.5+0.5)*255
                target_faces = (target_faces*0.5+0.5)*255
                ssim_scores.append(ssim(source_faces, fake_afters, data_range=255.0).item())
                psnr_socres.append(psnr(source_faces, fake_afters).item())
                mse_scores.append(F.mse_loss(source_faces, fake_afters).item())
                # 提取人脸嵌入
                emb_source = model.forward(F.interpolate(source_faces, size=img_shape, mode='bilinear'))
                emb_fake_after = model.forward(F.interpolate(fake_afters, size=img_shape, mode='bilinear'))
                emb_target = model.forward(F.interpolate(target_faces, size=img_shape, mode='bilinear'))
                # 余弦相似度评估
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
            print(model_name, " 攻击前 success rate:%f" % src_success_rate)
            print(model_name, " 平均 ssim:%f" % np.mean(ssim_scores))
            print(model_name, " 平均 psnr:%f" % np.mean(psnr_socres))
            print(model_name, " 平均 mse:%f" % np.mean(mse_scores))
            print(model_name, " 攻击后 success rate:%f" % rob_success_rate)
            print(model_name, " 攻击 success rate:%f" % (src_success_rate-rob_success_rate))
        print("---------------------------------------------------------")
        print("-----------------------模型测试结束------------------------")
        print("---------------------------------------------------------")

    @torch.no_grad()
    def generate_fake(self, source_img_path, target_img_path=None):
        # 加载测试模型参数
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
        # 加载配置文件中指定的预训练模型参数
        model_generator_dict = torch.load(model_dir + '/' + '%05d_generator.pth' % epoch_id)
        model_discriminator_dict = torch.load(model_dir + '/' + '%05d_discriminator.pth' % epoch_id)
        self.generator.load_state_dict(model_generator_dict)
        self.discriminator.load_state_dict(model_discriminator_dict)
