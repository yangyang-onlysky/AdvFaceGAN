# AdvFaceGAN
This project is a paper related code, training  GAN, generating adversarial face to attack FR models, and has achieved good attack effects on both offline models and commercial face comparison apis (Tencent, Aliyun and Face++). Currently under review for submission in peerj journal.

## 1. prepare environment

Suggestion use PyCharm IDE，and use shell bellow to set conda repository for avoid probably bugs:

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
```

please ensure your computer has installed cuda、cudnn、python3 and conda.

The following shell installs the conda environment "frb" for preprocessing original datasets(such as [casia-webface](https://www.kaggle.com/datasets/ntl0601/casia-webface)or[lfw](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz)) with 1_prepDataset.ipynb(**We suggests you download our preprocessed Datasets in "2. Download Everything You Need",or must install frb environment and preprocessing original datasets with Windows operating system,Because you may cann't install tensorflow-gpu 2.6 with Linux!!!**):

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

The following shell installs the conda environment "AdvFaceGAN" for training and testing your result models:

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

## 2. Download Everything You Need

The first is the substitute white-box Fr models that the training process needs :：[ckpts](https://drive.google.com/file/d/1l7tvppBVQfp2ZPiq-EYQ59bMtMaajTA3/view?usp=drive_link) ，This is required by starting the new training process, please download and unzip it to the "./fr_models" directory.

You can download our Datasets from these links:

Dataset casia-webface: https://figshare.com/articles/dataset/casia-aligned-112x112/27073465?file=49308127

Dataset lfw: https://figshare.com/articles/dataset/lfw-aligned-112x112/27073438

Dataset ffhq: https://figshare.com/articles/dataset/ffhq-aligned-112x112/27073375

Dataset celeba-hq: https://figshare.com/articles/dataset/celeba_hq-aligned-112x112_zip/27073291

Then is the resulting model that the authors themselves trained using the code of this project, which can be used directly for testing：[save_dir](https://drive.google.com/file/d/1izxC23w_2beu7C-MF08uiwid_Bkb60bQ/view?usp=drive_link)，This is required to test the resulting model directly, so download and unzip it to the root of the project.

## 3. Start new training

Open the command line at the root of the project code, execute the following script to call train.py for start new training process,But don't forget to set your datasets' folder in the configuration file!

(The configuration policy for this project is to read the default configuration from the configuration file, and allow you to adjust the core training parameters using command line parameters(you can find these parameters's meaning from train.py's main function))

```
python train.py --config="configuration file path" --tms=["white-box models used to training",] --pert=upperlimit of pertubation --output="save dir" --stlossfactor=stloss's factor --maxssim=upperlimit of ssim

# such as：
# training    impersonation attack & pert=4 & with stloss & maxssim 0.97  result model：
python train.py --config="config/target.ini" --pert=4 --output="./save_dir/target 4 8白盒 奇怪ssim" --maxssim=0.97

```

After starting the training correctly, the following progress display will appear:

![QQ_1726830553089](https://github.com/user-attachments/assets/3e562e5f-7a65-41ee-8204-04c492366a6e)

## 4. Start your testing

Open the command line at the root of the project code, execute the following script to call train.py to start testing:

(The configuration policy for this project is to read the default configuration from the configuration file, and allow you to adjust the core training parameters using command line parameters)

```
python test.py --config="configuration file path" --model_path="directory of model's pth file" --epoch=target epoch

# such as：
# evaluation    impersonation attack & pert=4 & with stloss & epoch=990 model：
python test.py --config="config/target.ini" --model_path="./save_dir/target 4 8白盒 奇怪ssim 97ssim/model" --epoch=990
# evaluation    impersonation attack & pert=3 & without stloss & epoch=990 model：
python test.py --config="config/target.ini" --model_path="./save_dir/target 3 8白盒 无stloss/model" --epoch=990
# evaluation    impersonation attack & pert=3 & with stloss & epoch=850 model：
python test.py --config="config/target.ini" --model_path="./save_dir/target 3 8白盒 奇怪ssim/model" --epoch=850
```



Switch between the two test modes by modifying the comment in the following position in test.py:

![QQ_1726828270957](https://github.com/user-attachments/assets/34a6bad2-a478-46dc-916c-d119da82859f)

The start_testing function reads the test_dataset_dir dataset in the configuration file, randomly selects 6000 sets of non-human faces to generate antagonistic faces, tests all models in the test_model_name_list, And output PSNR, MSE, SSIM, impersonation attack success rate and other evaluation indicators. In the figure below, the mobileface model FAR before the impersonation attack is 0.9974493, the model FAR after the impersonation attack is 0.108122, so the success rate of the impersonation attack is 0.9974493-0.108122=0.889372.

![QQ_1726829742586](https://github.com/user-attachments/assets/c5ad22bc-db7e-47da-a0fc-c1ef6d99a61f)

The generate_fake function will generate a adversary face with the specified two images, source as the attacker and target as the victim, stored in the test folder at the root of the project.

![QQ_1726828955684](https://github.com/user-attachments/assets/c7c9e30a-8afd-4869-a2d1-18a2b0fe07e9)

## 5. evaluating commercial face API

Firstly,Configure Aliyun, Tencent Cloud or Face++'s API key and secret in your System Environment Variables:

"ALIBABA_CLOUD_ACCESS_KEY_ID","ALIBABA_CLOUD_ACCESS_KEY_SECRET"

"FACEPP_API_KEY","FACEPP_API_SECRET"

"TENCENTCLOUD_SECRET_ID","TENCENTCLOUD_SECRET_KEY"

Then using 2_eval_aliyun.ipynb for Aliyun API、3_eval_Tencent.ipynb for Tencent API and 4_eval_faceplusplus.ipynb for Face++ API

Or, You can also simply use the sample we generated in the "./test" folder, source.png is attacker face, target.png is victim face and so fake.png is the adversary face which will be judged as victim by Commercial face API!

Commercial Face API such here：[Aliyun](https://vision.aliyun.com/experience/detail?spm=a2cvz.27720474.J_9219321920.16.be705d53Ftk66m&tagName=facebody&children=CompareFace) [Tencent](https://cloud.tencent.com/product/facerecognition) [face++](https://www.faceplusplus.com.cn/face-comparing/)
