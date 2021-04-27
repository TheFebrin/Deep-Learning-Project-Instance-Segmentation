# Pet-Dataset-Instance-Segmentation
Oxford-IIIT Pet Dataset - Instance Segmentation

## Table of contents
* [Notebooks](#notebooks)
* [Quick start](#quick-start)
* [Useful commands](#useful-commands)
* [General info](#general-info)
* [Folder structure](#folder-structure)
* [Main Components](#main-components)
* [Status](#status)
* [Credits](#credits)


--------------
## Notebooks

1. Data analysis [here](https://github.com/TheFebrin/Deep-Learning-Project-Instance-Segmentation/blob/master/notebooks/Data_analysis.ipynb).
2. Convert a custom dataset to COCO format [here](https://github.com/TheFebrin/Deep-Learning-Project-Instance-Segmentation/blob/master/notebooks/Convert_to_COCO_format.ipynb).
3. When making a custom dataset check if your data is compatible with the COCO format [here](https://github.com/TheFebrin/Deep-Learning-Project-Instance-Segmentation/blob/master/notebooks/Pycoco-test.ipynb).
4. Create a config to train the model [here](https://github.com/TheFebrin/Deep-Learning-Project-Instance-Segmentation/blob/master/notebooks/Create_config.ipynb).
5. Demo training [here](https://github.com/TheFebrin/Deep-Learning-Project-Instance-Segmentation/blob/master/notebooks/Demo%20training.ipynb).


--------------
## Quick start

Nice virtualenv tutorial [here](https://computingforgeeks.com/fix-mkvirtualenv-command-not-found-ubuntu/)
```bash
pip3 install --upgrade pip
```

```bash
which python3.8
mkvirtualenv -p <path to python3> <name>
workon <name>
```

For example:

```bash
mkvirtualenv -p /usr/bin/python3.8 Febrin
workon Febrin
```

Install requirements

```bash
pip3 install -r requirements.txt
```

Download dataset and models
```bash
python download.py
```

Install [mmdetection](https://github.com/open-mmlab/mmdetection)
with [mmcv](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
```bash
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
```

In our case we use `torch 1.8.0` and `Cuda 11.1`
```bash
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.102.04   Driver Version: 450.102.04   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
```
Install correct pytorch
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

So this is matching mmcv
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

--------------
## Useful commands

To divide dataset into train/test/valid folders:
```bash
python dataset/divide_dataset.py --datapoints-path=../images --labels-path=../annotations/trimaps \
    --datapoints-extention=.jpg --labels-extention=.png --valid=True --train-ratio=0.7
```


--------------
## General info

Our goal is to create an instance segmentation model based on the Oxford-IIIT Pet Dataset.

Instance segmentation is a task in Computer Vision that aims to identify each instance of each object within the image at the pixel level. In our case, each image contains a single object, which means our goal is to predict which pixels belong to the object and which to the background as well as predict the object's class.

Our dataset contains ~200 images for each from 37 classes.

![image](/figures/class-distribution.png)

Near 2/3 of those classes are dog breeds, the rest of them are cat breeds.

![image](/figures/species-distribution.png)

The images have different sizes and rations, therefore they need to be resized. There's a high variance in background colors and the amount of light on the pictures.

![image](/figures/class-samples.png)

We will build our model using [MMDetection](https://github.com/open-mmlab/mmdetection) - an open-source object detection toolbox based on PyTorch. We will experiment with different types of backbones, for example, ResNet, ResNext, or VGG.


--------------
## Folder structure

```
├── /model                 - folder for keeping models
│   └── simple_cnn_model.py
│   └── download_model.py  - script for downloading dataset
│
├── /trainer               - training scripts
│   └── train.py
|   └── test.py
│   
├──  /mains                - main files responsible for the whole pipeline
│    └── main.py 
│ 
├──  /notebooks            - notebook files created for tests on Colaboratory
│    ├── /Convert_to_COCO_format.ipynb
│    ├── /Create_config.ipynb
│    ├── /Demo training.ipynb
│    ├── /Demo.ipynb
│    ├── /Pycoco-test.ipynb
│    └── /Data_analysis.ipynb
| 
├──  /dataset              - things related to the dataset
│    ├── /train            - train datapoints and labels
│    ├── /test             - test datapoints and labels
│    ├── /valid            - valid datapoints and labels
│    └── divide_dataset.py - script that divides the dataset into /train/test/valid
│
└── /utils 
     ├── logger.py
     └── ...
```

## Main Components

--------------
### Dataset

After running the script `/dataset/divide_dataset.py`,
the dataset will be prepared in the same folder.


--------------
### Comet.ml logger

This template also supports reporting to Comet.ml which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.

Add your API key [in the configuration file](configs/example.json#L9):


For example:  `"comet_api_key": "your key here"`

Here's how it looks after you start training:
<div align="center">

![image](/figures/comet2.png)

</div>

You can also link your Github repository to your comet.ml project for full version control.
[Here's a live page showing the example from this repo](https://www.comet.ml/gidim/tensorflow-project-template/caba580d8d1547ccaed982693a645507/chart)

--------------
## Status
Project started: _09.03.2021_

1. Week 1:
    * Created the repository
    * Analyzed the dataset
    * Investigated data storing options (GCS) and model training (GCP machine)
    * Created a Kanban board where we track the tasks and progress.

2. Week 2:
    * Downloaded the data and created a dataloader
    * Started to familiarize ourselves with mmdet environment
    * Tested CometML and played with it
    * Created requirements.txt

3. Week 3:
    * Created our own CometML logger hook and integrated it with their API
    * Created a proper DataLoader
    * Started writing a mmdet notebook that will transition to train.py soon
    * Read some papers i.e. Feature Pyramid Network

4. Week 4:
    * Validated our coco using: https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools
    * Cleaned the repo and README
    *  Train on the second model to show that COMET ML

5. Week 5:
    * Moved our code from Colab files to Python scripts
    * Created Docerfile to be used as an environment created inside Colab

Project ended: _27.04.2021_



--------------
## Credits

* [@mdragula](https://github.com/mdragula)
* [@MatMarkiewicz](https://github.com/MatMarkiewicz)
* [@TheFebrin](https://github.com/TheFebrin)
