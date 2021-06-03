# Get start for CAT
*NOTE*: We reimplement our method based on [Swin](https://github.com/microsoft/Swin-Transformer), the models and logs is old version. You will run into some problems with the wrong module names, but it can be fixed manually. We will update these resources when we have time. But you can reproduce our work and results with the following instructions.

## STARTED

Clone the repository firstly:
```
git clone https://github.com/linhezheng19/CAT.git
cd CAT
```

### Classification

For classification, we need `pytorch` and `timm`:

```
conda create -n cat python=3.7
conda activate cat
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

Install other requirements:

```
pip install timm==0.3.2 opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

Install `Apex`:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
*NOTE*: You may install `Apex` failed, please run install as follows:
```
pip install -v --no-cache-dir ./
```

#### Data preparation

For standard ImageNet dataset, you can download it from [ImageNet](http://image-net.org/).

The file structure should as follows:
  ```
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...

  ```

#### Training from scratch

You can simplely run as follows:
```
python -m torch.distributed.launch --nproc_per_node <number-of-gpus> --master_port 10086 main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size <batch-size>
```
For `tiny`:
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 10086  main.py \
--cfg configs/cat_tiny.yaml --data-path data/CLS-LOC --batch-size 128
``` 

#### Evaluation

You can evaluate models as follows:
```
python -m torch.distributed.launch --nproc_per_node <number-of-gpus> --master_port 10086 main.py \
--eval --cfg <config-file> --resume <checkpoint-file> --data-path <imagenet-path>
```

### Detection

Out implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please install [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)

#### Training

To train CAT based detection methods, run as follows:
```
cd detection
```
Run `RetinaNet` with 8 gpus:
```
bash dist_train.sh configs/retinanet_cat_small_fpn_1x_coco.py 8 --options model.pretrained=<pretrained-model>
```

#### Evaluation

To evaluate the mAP of CAT based `RetinaNet` on COCO with 8 gpus, run:

```
bash dist_test.sh configs/retinanet_cat_small_fpn_1x_coco.py <checkpoint-file> 8 --eval mAP
```

### Segmentation

Out implementation is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Please install [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md)

#### Training

To train CAT based segmentation methods, run as follows:
```
cd segmentation
```
Run `Semantic FPN` with 8 gpus:
```
bash dist_train.sh configs/semantic_fpn_cat_small_512x512_80k_ade20k.py 8 --options model.pretrained=<pretrained-model>
```

#### Evaluation

To evaluate the mAP of CAT based `Semantic FPN` on COCO with 8 gpus, run:

```
bash dist_test.sh configs/semantic_fpn_cat_small_512x512_80k_ade20k.py <checkpoint-file> 8 --eval mIoU
```

### FLOPs

To evaluate FLOPs of methods:

```
cd detection # or cd segmentation
python get_flops.py <config-file> --shape <evaluate-shape>
```
