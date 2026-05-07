# PriMAT: Robust Multi-Animal Tracking for Primates in the Wild

<p align="center">
  <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0347669"><img alt="Paper" src="https://img.shields.io/badge/Paper-PLOS%20ONE-blue"></a>
  <a href="https://colab.research.google.com/github/ecker-lab/PriMAT-tracking/blob/main/notebooks/PriMAT_demo.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center">
  Bounding-box-based multi-animal tracking for challenging videos of non-human primates in the wild.
</p>

---

## Overview

PriMAT is a robust multi-animal tracking framework specifically adapted to challenges introduced by videos of non-human primates in the wild.

In addition to tracking, the framework can also be used to learn a classification task for each bounding box, for example **individual identification**.

---

## Paper

📄 **Read the paper here:**  
[PriMAT: Robust multi-animal tracking of primates in the wild](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0347669)

### Citation

```bibtex
@article{vogg2026primat,
  title={PriMAT: Robust multi-animal tracking of primates in the wild},
  author={Vogg, Richard and Nuske, Matthias and Weis, Marissa A and L{\"u}ddecke, Timo and Karako{\c{c}}, Elif and Ahmed, Zurna and Pereira, Sofia M and Malaivijitnond, Suchinda and Meesawat, Suthirote and Murphy, Derek and others},
  journal={PLoS One},
  volume={21},
  number={4},
  pages={e0347669},
  year={2026},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

---

## Examples

### Tracking macaques
![Tracking macaques](material/vid_630.gif)

### Tracking lemurs
![Tracking lemurs](material/Eval10.gif)

### Comparison with a keypoint-based model
![Keypoint tracking lemurs](material/Eval10_DLCfull.gif)

### Tracking and identifying lemurs
![Individual ID](material/e7_c4_identification.gif)

---

## Try it out

Train the tracking model and apply it to a video directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecker-lab/PriMAT-tracking/blob/main/notebooks/PriMAT_demo.ipynb)

---

## Setup

> **Important:** If you have more than one GPU in your system, you need to export/select one that is supported by **CUDA 10.2** for the entire setup.

Clone this repository and navigate to the project folder, then create the environment:

```bash
conda env create -f environment.yml
conda activate primat
```

If you want to create output videos, you also need to install [ffmpeg](https://www.ffmpeg.org/):

```bash
sudo apt install ffmpeg
```

---

## Data preparation

For training, the dataset should be organized in a folder with two subfolders: `images` and `labels_with_ids`.

```text
dataset_folder
├── images
│   ├── IMG0001.jpg
│   ├── IMG0002.jpg
│   └── ...
└── labels_with_ids
```

Use the notebook `notebooks/create_labels.ipynb` to convert your **VoTT** or **CVAT** annotations into the required format.

If your labels are stored in another format, either:
- adapt the data loader, or
- convert them into the formats described below.

---

### Tracking

If you only want to train the tracking model, you need one `.txt` file for each image in the `images` folder. Each label file must have the same name as the image file:

- `IMG0001.jpg` → `IMG0001.txt`

Each line in the file corresponds to one detection and contains:

```text
class id x_center y_center w h
```

The last four values are normalized by image width and image height.  
If you only have one object type, `class` can always be `0`.

In our setup:
- `0` = lemur
- `1` = feeding box

The `id` is a running number through the whole dataset, i.e. each individual in each image receives a new ID.

Example:

```text
0 379 0.520337 0.464083 0.031490 0.041500
0 380 0.538707 0.470500 0.024742 0.041333
1 337 0.547142 0.462833 0.020993 0.028000
1 338 0.498782 0.518167 0.033739 0.056000
```

---

### Individual identification

For individual identification, or other classification tasks, an additional class label is appended to each row.

Format:

```text
class id x_center y_center w h classification_label
```

The first six values are the same as above.

In our experiments:
- `7` was used as an **"Unsure"** class
- only lemurs were classified
- feeding boxes were assigned class `7` and ignored during training

Example:

```text
0 65 0.682461 0.544545 0.287594 0.446381 7
0 51 0.983571 0.44805 0.048391 0.120989 2
0 70 0.197715 0.476319 0.167154 0.230225 5
1 14 0.141534 0.510369 0.141242 0.207826 7
1 17 0.52678 0.527526 0.164393 0.311018 7
```

---

## Pretrained models and datasets

📦 **Download pretrained weights and datasets used in the publication:**  
[https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/CMQY0Q](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/CMQY0Q)

### Available pretrained and baseline models
- ImageNet pretrained model
- MacaqueCopyPaste pretrained model
- Macaque tracking model
- Lemur and box tracking model
- Lemur identification model

### Macaque tracking and evaluation
- 500 macaque images for training
- 12 macaque videos for evaluation

### Lemur tracking and evaluation
- 500 lemur images for training
- 12 lemur videos for evaluation

### Lemur identification and evaluation
- Lemur ID images for training
- 30 lemur ID videos for evaluation

---

## Train on a custom dataset

You can train the tracking model on your own dataset using the following steps:

1. Generate one `.txt` label file per image.  
   Each line should follow the format:

   ```text
   class id x_center/img_width y_center/img_height w/img_width h/img_height
   ```

   See `notebooks/create_labels.ipynb` for code compatible with **VoTT** and **CVAT** annotations.

2. Generate files containing image paths.  
   Example files can be found in `src/data/`.  
   See `notebooks/create_labels_own.ipynb` for an example.

3. Create a `.json` file for your custom dataset in `src/lib/cfg/`.  
   At minimum, specify the keys:
   - `"root"`
   - `"train"`

   Examples are available in `src/lib/cfg/`.

4. Add the following argument to your experiment file during training:

   ```bash
   --data_cfg '../src/lib/cfg/your_dataset.json'
   ```

---

## Inference and video output

Use `experiments/run_inference.py` and adapt it to your videos and models.

If you set:

```bash
--output_format video
```

a video showing the tracking results will be generated using `ffmpeg`.

---

## Evaluation

### Tracking evaluation
To evaluate tracking results using **HOTA**, clone the [TrackEval](https://github.com/JonathonLuiten/TrackEval) repository.

Then use:

```text
notebooks/evaluate_tracking_HOTA.ipynb
```

to move files into the correct locations and evaluate the model.

### Identification evaluation
For evaluating identification results, use:

```text
notebooks/evaluate_lemur_id_on_videos.ipynb
```

---

## Acknowledgement

We used the codebase from **FairMOT** as a starting point:

> [**FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking**](http://arxiv.org/abs/2004.01888)  
> Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, Wenyu Liu

We also used multi-class modifications from:

> [**CaptainEven**](https://github.com/CaptainEven/MCMOT)

A large part of their code builds on:
- [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT)
- [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)

Thanks to the authors for their excellent work.
