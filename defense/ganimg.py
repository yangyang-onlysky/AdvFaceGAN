import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from defense.AdvFaceGAN import Generator
from fr_models.config import threshold_lfw
from fr_models.get_model import getmodel
from torchvision.utils import save_image

def load_img(img_path):
    # 加载输入图像
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])
    image_tensor = transform(image).unsqueeze(0).cuda()  # 增加batch维度并转到指定设备
    return image_tensor

defense_generator = Generator().to("cuda")
defense_generator.load_state_dict(torch.load("defense/00940_generator.pth", map_location="cuda"))
defense_generator.eval()

# 生成一张并保存
_, clean_target_tensor = defense_generator(load_img("defense/fake.png"))
save_image(clean_target_tensor*0.5+0.5, "defense/clean.png")

# 批量评测
# fake_dir = "data/AdvFaceGAN_target 5 8白盒 奇怪ssim 92ssim 双身份损失0.15 2490_lfw_eps5_tpert4.6"
# fr_model_name = "FaceNet-casia"
# frmodel,img_shape = getmodel(fr_model_name)
# th = threshold_lfw[fr_model_name]['cos']
#
# FAR_simi_scores = []
# source_simi_scores = []
# target_simi_scores = []
# fake_pert=[]
#
# clean_source_pert=[]
# clean_target_pert=[]
# after_FAR_simi_scores = []
# after_TAR_simi_scores = []
# after_source_simi_scores = []
# after_target_simi_scores = []
# # 使用 os.walk 遍历目录及子目录
# for dirpath, dirnames, filenames in os.walk(fake_dir):
#     for dirname in tqdm(dirnames):
#         fake_pair_dir = os.path.join(fake_dir, dirname)
#         fake_img_tensor = load_img(os.path.join(fake_pair_dir, "adv.png"))
#         with torch.no_grad():  # 禁用梯度计算
#             _, clean_fake_tensor = defense_generator(fake_img_tensor)
#             source_img_tensor = load_img(os.path.join(fake_pair_dir, "source.png"))
#             target_img_tensor = load_img(os.path.join(fake_pair_dir, "target.png"))
#             _, clean_source_tensor = defense_generator(source_img_tensor)
#             _, clean_target_tensor = defense_generator(target_img_tensor)
#             # extract face embedding
#             emb_source = frmodel.forward(F.interpolate((source_img_tensor*0.5+0.5) * 255, size=img_shape, mode='bilinear'))
#             emb_target = frmodel.forward(F.interpolate((target_img_tensor*0.5+0.5) * 255, size=img_shape, mode='bilinear'))
#             emb_fake_after = frmodel.forward(F.interpolate((fake_img_tensor*0.5+0.5) * 255, size=img_shape, mode='bilinear'))
#             emb_clean_fake = frmodel.forward(F.interpolate((clean_fake_tensor*0.5+0.5) * 255, size=img_shape, mode='bilinear'))
#             emb_clean_source = frmodel.forward(F.interpolate((clean_source_tensor*0.5+0.5) * 255, size=img_shape, mode='bilinear'))
#         #  evaluation cosine similarity
#         FAR_simi_scores.extend(torch.cosine_similarity(emb_source, emb_target).tolist())
#         source_simi_scores.extend(torch.cosine_similarity(emb_fake_after, emb_source).tolist())
#         target_simi_scores.extend(torch.cosine_similarity(emb_fake_after, emb_target).tolist())
#         fake_pert.append(torch.norm(
#             F.interpolate((clean_fake_tensor*0.5+0.5), size=(112, 112), mode='bilinear') - F.interpolate((fake_img_tensor*0.5+0.5), size=(112, 112),
#                                                                                         mode='bilinear')).item())
#
#         clean_source_pert.append(torch.norm(
#             F.interpolate((clean_source_tensor*0.5+0.5), size=(112, 112), mode='bilinear') - F.interpolate((source_img_tensor*0.5+0.5), size=(112, 112),
#                                                                                         mode='bilinear')).item())
#         clean_target_pert.append(torch.norm(
#             F.interpolate((clean_target_tensor * 0.5 + 0.5), size=(112, 112), mode='bilinear') - F.interpolate(
#                 (target_img_tensor * 0.5 + 0.5), size=(112, 112),
#                 mode='bilinear')).item())
#         after_FAR_simi_scores.extend(torch.cosine_similarity(emb_clean_fake, emb_target).tolist())
#         after_TAR_simi_scores.extend(torch.cosine_similarity(emb_clean_source, emb_source).tolist())
#         after_source_simi_scores.extend(torch.cosine_similarity(emb_clean_fake, emb_source).tolist())
#         after_target_simi_scores.extend(torch.cosine_similarity(emb_clean_fake, emb_target).tolist())
# print(fr_model_name, " before FAR:%f" % np.mean(np.array(FAR_simi_scores)>th))
# print(fr_model_name, " attack success rate1:%f" % np.mean(np.array(target_simi_scores) > th))
# print(fr_model_name, " attack success rate2:%f" % np.mean((np.array(source_simi_scores) > th) & (np.array(target_simi_scores) > th)))
# print(fr_model_name, " FSS&FTS:%f & %f" % (np.mean(np.array(source_simi_scores)),np.mean(np.array(target_simi_scores))))
#
# print(fr_model_name, " fake_pert:%f" % np.mean(np.array(fake_pert)))
#
# print(fr_model_name, " clean_source_pert:%f" % np.mean(np.array(clean_source_pert)))
# print(fr_model_name, " clean_target_pert:%f" % np.mean(np.array(clean_target_pert)))
# print(fr_model_name, " after FAR:%f" % np.mean(np.array(after_FAR_simi_scores)>th))
# print(fr_model_name, " after clean source TAR:%f" % np.mean(np.array(after_TAR_simi_scores) > th))
# print(fr_model_name, " after TAR:%f" % np.mean(np.array(after_source_simi_scores) > th))
# print(fr_model_name, " after attack success rate1:%f" % np.mean(np.array(after_target_simi_scores) > th))
# print(fr_model_name, " after attack success rate2:%f" % np.mean((np.array(after_source_simi_scores) > th) & (np.array(after_target_simi_scores) > th)))
# print(fr_model_name, " FSS&FTS:%f & %f" % (np.mean(np.array(after_source_simi_scores)),np.mean(np.array(after_target_simi_scores))))