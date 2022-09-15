# LayoutAnalysis

## 환경설정(pytorch cuda 11.6)
```

docker run --name [NAME] -it -v [HOSTPATH]:/opt --gpus all python:latest

exit()

docker start [NAME]

docker exec -it [NAME] /bin/bash

wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

bash Anaconda3-2019.10-Linux-x86_64.sh

source ~/.bashrc

conda create -n py3.8 python==3.8

conda activate py3.8

cd opt

git clone https://github.com/JoonHyun814/LayoutAnalysis.git
```

## 데이터 셋 download
```
wget https://guillaumejaume.github.io/FUNSD/dataset.zip

unzip dataset.zip

mv dataset FUNSD
```
## 필요 패키지 설치
```
cd jejedu_LayoutAnalysis

pip install -r requirements.txt

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## pretrained model download

east_vgg16.pth download
https://drive.google.com/file/d/1gkdKFrIYp_T9K2fziBTaypyuuDD5VRgu/view?usp=sharing

vgg16_bn-6c64b313.pth download
https://drive.google.com/file/d/1wetfUNTHO_2aKfRRTmtylY2bAV5D9CjC/view?usp=sharing

jejedu_LayoutAnalysis/pths/ 에 저장
