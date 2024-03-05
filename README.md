# Multi Object Tracking with Monkeys

## Setup

- IMPORTANT NOTICE: if you have more than one GPU in your system you have to export one thats supported by cuda 10.2 for the entirety of the setup!
- Install dependencies. We use python 3.8 and pytorch >= 1.7.0 also for DCNv2 gcc and g++ < 9 are needed (thus they are included in the environment.yaml file).

```
conda env create -f environment.yaml
conda activate mktrack
cd {path to project folder}
```

Currently, it seems that not all the packages are correctly installed in the environment file. So when you run the first training or inference, it might be needed to install a few packages manually. This will be fixed.

- If you plan to create output videos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

```
sudo apt install ffmpeg
```

## Data preparation

In general we would like to have the data for training in one folder. This folder has two subfolders, images and labels_with_ids.

```
dataset_folder
   |——————images
   |        └——————IMG0001.jpg
   |        └——————IMG0002.jpg
   |        ...
   └——————labels_with_ids
```         

## Pretrained models and baseline model

Send me a message if you need well working models for Macaques or Lemurs.


## Train on custom dataset

You can train the tracking model on custom datasets by following several steps below:

1. Generate one txt label file for one image. Each line of the txt label file represents one object. The format of the line is: "class id x_center/img_width y_center/img_height w/img_width h/img_height". Check notebooks/create_labels_own.ipynb to get code working for data labelled with VoTT and CVAT.
2. Generate files containing image paths. The example files are in src/data/. Check notebooks/create_labels_own.ipynb for creating this file.
3. Create a json file for your custom dataset in src/lib/cfg/. You need to specify the "root" and "train" keys in the json file. You can find some examples in src/lib/cfg/.
4. Add --data_cfg '../src/lib/cfg/your_dataset.json' to your experiment file when training.

## Inference and video output

Use the file in experiments/run_inference.py and adapt it to your videos and models.
If you set --output_format to "video" in this file, a video of the tracking results will be generated (using ffmpeg).

## Acknowledgement

Using codebase and ideas from FairMOT:

> [**FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking**](http://arxiv.org/abs/2004.01888),  
> Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, Wenyu Liu,  
> _arXiv technical report ([arXiv 2004.01888](http://arxiv.org/abs/2004.01888))_

As well as multi-class alterations from:

> [**CaptainEven**](https://github.com/CaptainEven/MCMOT)

A large part of their code is borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet). Thanks for their wonderful works.

