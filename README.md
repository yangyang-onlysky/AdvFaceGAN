# AdvFaceGAN
此项目是论文相关代码，训练生成对抗网络，用于生成对抗人脸攻击人脸识别系统，在离线开源模型和商业人脸比对API（Tencent、Aliyun和Face++）上都取得了良好的攻击效果。
当前正在peerj期刊投稿审核中。

## 1. 环境准备

建议IDE使用pycahrm

安装python3、安装conda环境（此处忽略教程）

以下命令安装用于预处理数据的conda环境frb：

```shell
conda deactivate
conda remove -n frb --all -y
conda create -n frb -y
conda activate frb
conda install tensorflow-gpu==2.6.0 -y
conda install -c fastai opencv-python-headless -y
conda install imageio -y
conda install numpy==1.23.4 -y
conda install scikit-image -y
conda install tqdm -y
conda install ipykernel -y
python -m ipykernel install --user --name=frb --display-name "frb"
```

以下命令安装用于训练模型与测试的conda环境AdvFaceGAN：

```shell
conda deactivate
conda remove -n AdvFaceGAN --all -y
conda create -n AdvFaceGAN -y
conda activate AdvFaceGAN
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install torchmetrics tqdm tensorboard multiprocess  -y
conda install -c conda-forge jupyter -y
conda install matplotlib -y
conda install ipykernel -y
python -m ipykernel install --user --name=AdvFaceGAN --display-name "AdvFaceGAN"

pip install alibabacloud_facebody20191230
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
```

## 2. 下载预训练模型

首选是训练模型需要使用到的替代预训练人脸识别模型：[ckpts](https://drive.google.com/file/d/1l7tvppBVQfp2ZPiq-EYQ59bMtMaajTA3/view?usp=drive_link) ，这是开始新的训练必须下载的，请下载并解压到fr_models目录下。

然后是作者自己使用本项目代码训练得到的结果模型，可直接用于测试：[save_dir](https://drive.google.com/file/d/1izxC23w_2beu7C-MF08uiwid_Bkb60bQ/view?usp=drive_link)，这是想直接测试结果模型必须下载的，请下载并解压到项目根目录下。

## 3. 开始新的训练

在项目代码根目录打开命令行，并执行如下脚本调用train.py开始训练：

（本项目的配置策略为：从配置文件读取默认配置，并允许使用命令行参数调整核心训练参数）

```
python train.py --config=配置文件路径 --tms=训练所用所有白盒模型 --pert=扰动上限 --output=训练成果保存目录 --stlossfactor=stloss的因子 --maxssim=ssim上限

例如：
训练 以配置文件中默认的train_model_name_list为替代模型、扰动上限为4、ssim下限为0.97的模型
python train.py --config="config/target.ini" --pert=4 --output="./save_dir/target 4 8白盒 奇怪ssim" --maxssim=0.97

```

正确开始训练后会出现如下图的进度显示：

![QQ_1726830553089](md_pic\QQ_1726830553089.png)

## 4. 测试结果模型

可通过修改test.py中下图位置的注释，切换两种测试模式：

![QQ_1726828270957](md_pic\QQ_1726828270957.png)

start_testing会使用配置文件中的test_dataset_dir数据集，随机选择6000组非同人人脸生成对抗人脸，测试test_model_name_list中的所有模型，并输出PSNR、MSE、SSIM以及冒充攻击成功率等各项评估指标。如下图冒充攻击前mobileface模型FAR为0.9974493，冒充攻击后模型FAR为0.108122，冒充攻击成功率为0.889372。

![QQ_1726829742586](md_pic\QQ_1726829742586.png)

generate_fake会以指定的两张图片，source作为攻击者，target作为受害者，生成对抗人脸，保存于项目根目录的test文件夹。

![QQ_1726828955684](md_pic\QQ_1726828955684.png)

在项目代码根目录打开命令行，并执行如下脚本调用test.py开始测试：

（本项目的配置策略为：从配置文件读取默认配置，并允许使用命令行参数调整核心测试参数）

```
python test.py --config="config/target.ini" --model_path=模型pth文件所在目录 --epoch=轮数

例如：
测试 扰动量为4且加入stloss的冒充攻击第990轮训练的模型：
python test.py --config="config/target.ini" --model_path="./save_dir/target 4 8白盒 奇怪ssim 97ssim/model" --epoch=990
测试 扰动量为3且不加入stloss的冒充攻击第990轮训练的模型：
python test.py --config="config/target.ini" --model_path="./save_dir/target 3 8白盒 无stloss/model" --epoch=990
测试 扰动量为3且加入stloss的冒充攻击第850轮训练的模型：
python test.py --config="config/target.ini" --model_path="./save_dir/target 3 8白盒 奇怪ssim/model" --epoch=850
```

