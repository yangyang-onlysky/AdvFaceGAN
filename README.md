# AdvFaceGAN
此项目是一论文代码，训练生成对抗网络，用于生成对抗人脸攻击人脸识别系统，在离线开源模型和商业人脸比对API（Tencent、Aliyun和Face++）上都取得了良好的攻击效果。
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

首选是训练模型需要使用到的替代预训练人脸识别模型：[ckpts]() ,下载并解压到fr_models目录下。

然后是作者使用本项目代码训练得到的结果模型，可直接用于测试：[save_dir]()，下载并解压到项目根目录下。

## 3. 开始新的训练

在项目代码根目录打开命令行，并执行如下脚本调用train.py开始训练：

（本项目的配置策略为：从配置文件读取默认配置，并允许使用命令行参数调整核心训练参数）

```
python train.py --config="config/target.ini" 
```



## 4. 测试结果模型

在项目代码根目录打开命令行，并执行如下脚本调用test.py开始测试：

（本项目的配置策略为：从配置文件读取默认配置，并允许使用命令行参数调整核心测试参数）

```
python test.py --config="config/target.ini" --model_path=模型pth文件所在目录 --epoch=轮数

例如：
python test.py --config="config/target.ini" --model_path="./save_dir/target 3 8白盒 无stloss/model" --epoch=990
python test.py --config="config/target.ini" --model_path="./save_dir/target 3 8白盒 奇怪ssim/model" --epoch=850
```

