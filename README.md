# PriMAT: A robust multi-animal tracking model for primates in the wild

## Link to our paper
when it is published

## Idea
We present a bounding box based multi-animal tracking model, which was specifically adapted for challenges introduced by videos of non-human primates in the wild.

- Tracking macaques
![Tracking macaques](material/vid_630.gif)

- Tracking lemurs
![Tracking lemurs](material/Eval10.gif)

- Comparison with keypoint based model
![Keypoint tracking lemurs](material/Eval10_DLCfull.gif)

It can also be used to learn a classification task for each bounding box, e.g. for individual identification.
- Tracking and identifying lemurs
![Individual ID](material/e7_c4_identification.gif)

## Try it out

Train the tracking model and apply to a video:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecker-lab/PriMAT-tracking/blob/main/notebooks/PriMAT_demo.ipynb)

## Setup

- IMPORTANT NOTICE: if you have more than one GPU in your system you have to export one thats supported by cuda 10.2 for the entirety of the setup!


Clone this repository and navigate to the folder it is saved in.

```
conda env create -f environment.yml
conda activate primat
```

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

Use the notebook in notebooks/create_labels.ipynb to bring your VoTT or CVAT annotations in the right format. If your labels are in a different format, either change the data loader or bring them into the following format (depending on the task).

### Tracking
If you only want to train the tracking model, you will need to have one .txt file for each image in the images folder, having the same name as the image file (i.e. IMG0001.jpg will have a corresponding IMG0001.txt file in the labels_with_ids folder). The file consists of one line for each detection and six values separated by an empty space (see below). The values belong to class id x_center y_center w h. The last four values are normalized by image_width and image_height respectively. If you only have one type of objects in the image class can always be 0. We use 0 for lemurs and 1 for feeding boxes. The id is just a running number through the whole dataset (each individual on each image gets a new ID).

```
0 379 0.520337 0.464083 0.031490 0.041500
0 380 0.538707 0.470500 0.024742 0.041333
1 337 0.547142 0.462833 0.020993 0.028000
1 338 0.498782 0.518167 0.033739 0.056000
```

### Individual identification
For individual identification (or other classification tasks), the class of each row has to be added in the end (the first six values are the same as above). We used 7 as "Unsure" class, and were only interested in classifying lemurs. The boxes just received 7s, which were ignored for training.

```
0 65 0.682461 0.544545 0.287594 0.446381 7
0 51 0.983571 0.44805 0.048391 0.120989 2
0 70 0.197715 0.476319 0.167154 0.230225 5
1 14 0.141534 0.510369 0.141242 0.207826 7
1 17 0.52678 0.527526 0.164393 0.311018 7
```

## Pretrained models and datasets

[Download here](https://owncloud.gwdg.de/index.php/s/Mq4m9k1B74cN6ys) model weights and datasets used in this publication.

### Pretrained models and baseline models
- Imagenet pretrained model
- MacaqueCopyPaste pretrained model
- Macaque tracking model
- Lemur and box tracking model
- Lemur identification model

### Macaque tracking and evaluation
- 500 macaque images for training
- 12 macaques videos for evaluation

### Lemur tracking and evaluation
- 500 lemur images for training
- 12 lemur videos for evaluation

### Lemur identification and evaluation
- lemur id images for training
- 30 lemur id videos for evaluation

## Train on custom dataset

You can train the tracking model on custom datasets by following several steps below:

1. Generate one txt label file for one image. Each line of the txt label file represents one object. The format of the line is: "class id x_center/img_width y_center/img_height w/img_width h/img_height". Check notebooks/create_labels.ipynb to get code working for data labelled with VoTT and CVAT.
2. Generate files containing image paths. The example files are in src/data/. Check notebooks/create_labels_own.ipynb for creating this file.
3. Create a json file for your custom dataset in src/lib/cfg/. You need to specify the "root" and "train" keys in the json file. You can find some examples in src/lib/cfg/.
4. Add --data_cfg '../src/lib/cfg/your_dataset.json' to your experiment file when training.

## Inference and video output

Use the file in experiments/run_inference.py and adapt it to your videos and models.
If you set --output_format to "video" in this file, a video of the tracking results will be generated (using ffmpeg).

## Evaluation

For evaluating your tracking results using HOTA, you will need to clone the [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository. Then notebooks/evaluate_tracking_HOTA.ipynb will guide you to move all files to the correct position and evaluate your model.
For evaluating your identification results, please directly use notebooks/evaluate_lemur_id_on_videos.ipynb.

## Acknowledgement

We used the codebase from FairMOT as a starting point:

> [**FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking**](http://arxiv.org/abs/2004.01888),  
> Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, Wenyu Liu,  

As well as multi-class alterations from:
> [**CaptainEven**](https://github.com/CaptainEven/MCMOT)

A large part of their code builds on [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet). Thanks for their wonderful works.

