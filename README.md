# [AAAI2024] Spatial Transform Decoupling for Oriented Object Detection

Full paper is available at https://arxiv.org/abs/2308.10561.

## Results and models

All models, logs and submissions is available at [pan.baidu.com](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg).

> Password of `pan.baidu.com`: STDC

__All models can be downloaded in release mode now!__

Imagenet MAE pre-trained ViT-S backbone: [mae_vit_small_800e.pth](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/mae_vit_small_800e.pth)

Imagenet MAE pre-trained ViT-B backbone: [mae_pretrain_vit_base_full.pth](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/mae_pretrain_vit_base_full.pth) or [official MAE weight](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth)

Imagenet MAE pre-trained HiViT-B backbone: [mae_hivit_base_dec512d8b_hifeat_p1600lr10.pth](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/mae_hivit_base_dec512d8b_hifeat_p1600lr10.pth)

DOTA-v1.0 (multi-scale)

|               Model                |  mAP  | Angle | lr schd | Batch Size | Configs | Models |  Logs  | Submissions |
| :--------------------------------: | :---: | :---: | :-----: | :--------: | :-----: | :----: | :----: | :---------: |
|  STD with Oriented RCNN and ViT-B  | 81.66 | le90  |   1x    |    1\*8    | [cfg](./mmrotate-main/configs/rotated_imted/dota/vit/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_dota_ms_rr_le90_stdc_xyawh321v.py) | [model](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/orcnn_std_vit_dota_epoch_12.pth) | [log](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/orcnn_std_vit_dota_20240328_185845.log) | [submission](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/ms_ovs8.zip) |
| STD with Oriented RCNN and HiViT-B | 82.24 | le90  |   1x    |    1\*8    | [cfg](./mmrotate-main/configs/rotated_imted/dota/hivit/rotated_imted_hb1m_oriented_rcnn_hivitdet_base_1x_dota_ms_rr_le90_stdc_xyawh321v.py) | [model](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/orcnn_std_hivit_dota_epoch_12.pth) | [log](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/orcnn_std_hivit_dota_20230805_184646.log) | [submission](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/ms_ohs8.zip) |

HRSC2016

|               Model                | mAP(07) | mAP(12) | Angle | lr schd | Batch Size | Configs | Models |  Logs  |
| :--------------------------------: | :-----: | :-----: | :---: | :-----: | :--------: | :-----: | :----: | :----: |
|  STD with Oriented RCNN and ViT-B  |  90.67  |  98.55  | le90  |   3x    |    1\*8    | [cfg](./mmrotate-main/configs/rotated_imted/hrsc/vit/rotated_imted_oriented_rcnn_vit_base_3x_hrsc_rr_le90_stdc_xyawh321v.py) | [model](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/orcnn_std_vit_hrsc_epoch_36.pth) | [log](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/orcnn_std_vit_hrsc_20230814_214056.log) |
| STD with Oriented RCNN and HiViT-B |  90.63  |  98.20  | le90  |   3x    |    1\*8    | [cfg](./mmrotate-main/configs/rotated_imted/hrsc/hivit/rotated_imted_oriented_rcnn_hivitdet_base_3x_hrsc_rr_le90_stdc_xyawh321v.py) | [model](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/orcnn_std_hivit_hrsc_epoch_36.pth) | [log](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/releases/download/STD-240413/orcnn_std_hivit_hrsc_20230808_230504.log) |

## Installation - by docker

### 1. Build docker image
```cmd
docker build -t mmcv https://github.com/open-mmlab/mmcv.git#master:docker/release --build-arg MMCV=1.6.1 --build-arg PYTORCH=1.7.0 --build-arg CUDA=11.0 --build-arg CUDNN=8
```

The `mmcv` version and `pytorch` version are referred by the official paper and then, `cudnn` and `cuda` are referred by the official pytorch docker image according to the `pytorch` version

Or, you can just use the docker image I uploaded: `onedang2/mmcv:std`

### 2. Run docker image 
```cmd
docker tag mmcv:latest mmcv:1.6.1-torch1.7.0-cuda11.0-cudnn8
docker run -it -d --name std --gpus all --ipc host -v /HDD:/HDD mmcv:1.6.1-torch1.7.0-cuda11.0-cudnn8 bash
```

### 3. Install mmdet/mmrotate and DOTA_devkit
```cmd
pip install timm apex
pip install mmdet==2.25.1
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
cd ../
git clone https://github.com/yuhongtian17/Spatial-Transform-Decoupling.git
cp -r Spatial-Transform-Decoupling/mmrotate-main/* mmrotate/
```

If you want to conduct offline testing on the DOTA-v1.0 dataset (for example, our ablation study is trained on the train-set and tested on the val-set), we recommend using the official [DOTA devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit). Here we modify the evaluation code for ease of use.

```shell
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git
cd DOTA_devkit
sudo apt install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
cd ../
# 
git clone https://github.com/yuhongtian17/Spatial-Transform-Decoupling.git
cp Spatial-Transform-Decoupling/DOTA_devkit-master/dota_evaluation_task1.py DOTA_devkit/
```

Example usage:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup ./tools/dist_train.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 8 > nohup.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth 8 --eval mAP
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth 8 --format-only --eval-options submission_dir="./work_dirs/Task1_rotated_faster_rcnn_r50_fpn_1x_dota_le90_epoch_12/"
python "../DOTA_devkit/dota_evaluation_task1.py" --mergedir "./work_dirs/Task1_rotated_faster_rcnn_r50_fpn_1x_dota_le90_epoch_12/" --imagesetdir "./data/DOTA/val/" --use_07_metric True
```

## How to run

### 1. Prepare the dataset as DOTA format 

### - Download the dataset from [https://captain-whu.github.io/DOTA/dataset.html](https://captain-whu.github.io/DOTA/dataset.html)

#### - To crop the original images into 1024×1024 patches with an overlap of 200,
```shell
python src/tools/data/dota/split/img_split.py --base-json src/tools/data/dota/split/custom_split_configs/ss_train.json
python src/tools/data/dota/split/img_split.py --base-json src/tools/data/dota/split/custom_split_configs/ss_val.json
python src/tools/data/dota/split/img_split.py --base-json src/tools/data/dota/split/custom_split_configs/ss_test.json
```

> You must change original DOTA dataset directory in json files.

#### - To get a multiple scale crop image of dataset,
```shell
python src/tools/data/dota/split/img_split.py --base-json src/tools/data/dota/split/custom_split_configs/ms_train.json
python src/tools/data/dota/split/img_split.py --base-json src/tools/data/dota/split/custom_split_configs/ms_val.json
python src/tools/data/dota/split/img_split.py --base-json src/tools/data/dota/split/custom_split_configs/ms_test.json
```

> You must change original DOTA dataset directory in json files.

### 2. Train

#### rotated_faster_rcnn

```shell
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup ./tools/dist_train.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 8 > nohup.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth 8 --eval mAP
```

### 3. Test

#### rotated_faster_rcnn
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth 8 --format-only --eval-options submission_dir="./work_dirs/Task1_rotated_faster_rcnn_r50_fpn_1x_dota_le90_epoch_12/"
python "../DOTA_devkit/dota_evaluation_task1.py" --mergedir "./work_dirs/Task1_rotated_faster_rcnn_r50_fpn_1x_dota_le90_epoch_12/" --imagesetdir "./data/DOTA/val/" --use_07_metric True
```

## Acknowledgement

Please also support two representation learning works on which this work is based:

imTED: [paper](https://arxiv.org/abs/2205.09613) [code](https://github.com/LiewFeng/imTED)

HiViT: [paper](https://arxiv.org/abs/2205.14949) [code](https://github.com/zhangxiaosong18/hivit)

Also thanks to [Xue Yang](https://yangxue0827.github.io/) for his inspiration in the field of Oriented Object Detection.

## News

[VMamba](https://github.com/MzeroMiko/VMamba)-DOTA is available at [here](https://github.com/AkitsukiM/VMamba-DOTA)! A brand new model!

## Citation

```
@inproceedings{yu2024spatial,
  title={Spatial Transform Decoupling for Oriented Object Detection},
  author={Yu, Hongtian and Tian, Yunjie and Ye, Qixiang and Liu, Yunfan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={6782--6790},
  year={2024}
}
```

## License

STD is released under the [License](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/blob/main/LICENSE).
