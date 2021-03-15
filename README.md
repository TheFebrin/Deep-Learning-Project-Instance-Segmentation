# Pet-Dataset-Instance-Segmentation
Oxford-IIIT Pet Dataset - Instance Segmentation

## Table of contents
* [General info](#general-info)
* [Quick start](#quick-start)
* [Project architecture](#project-architecture)
* [Folder structure](#folder-structure)
* [Main Components](#main-components)
	-  [Models](#models)
	-  [Trainer](#trainer)
	-  [Data Loader](#data-loader)
	-  [Logger](#logger)
	-  [Configuration](#configuration)
	-  [Main](#main)
* [Status](#status)
* [Credits](#credits)

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
## Quick start
```
workon <Name>
```
```
mkvirtualenv -p /usr/local/bin/python3.8 <Name>
pip install -r requirements.txt
```

TODO

--------------
## Project architecture

<div align="center">

<img align="center" hight="600" width="600" src="https://github.com/Mrgemy95/Tensorflow-Project-Templete/blob/master/figures/diagram.png?raw=true">

</div>

--------------
## Folder structure

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains any model of your project.
│   └── example_model.py
│
│
├── trainer             - this folder contains trainers of your project.
│   └── example_trainer.py
│   
├──  mains              - here's the main(s) of your project (you may need more than one main).
│    └── example_main.py  - here's an example of main that is responsible for the whole pipeline.

│  
├──  data _loader  
│    └── data_generator.py  - here's the data_generator that is responsible for all data handling.
│
└── utils
     ├── logger.py
     └── any_other_utils_you_need

```

## Main Components

--------------
### Models
- #### **Base model**

    Base model is an abstract class that must be Inherited by any model you create, the idea behind this is that there's much shared stuff between all models.
    The base model contains:
    - ***Save*** -This function to save a checkpoint to the desk.
    - ***Load*** -This function to load a checkpoint from the desk.
    - ***Cur_epoch, Global_step counters*** -These variables to keep track of the current epoch and global step.
    - ***Init_Saver*** An abstract function to initialize the saver used for saving and loading the checkpoint, ***Note***: override this function in the model you want to implement.
    - ***Build_model*** Here's an abstract function to define the model, ***Note***: override this function in the model you want to implement.
- #### **Your model**
    Here's where you implement your model.
    So you should :
    - Create your model class and inherit the base_model class
    - override "build_model" where you write the tensorflow model you want
    - override "init_save" where you create a tensorflow saver to use it to save and load checkpoint
    - call the "build_model" and "init_saver" in the initializer.

--------------
### Trainer

- #### **Base trainer**
    Base trainer is an abstract class that just wrap the training process.

- #### **Your trainer**
     Here's what you should implement in your trainer.
    1. Create your trainer class and inherit the base_trainer class.
    2. override these two functions "train_step", "train_epoch" where you implement the training process of each step and each epoch.

--------------
### Data Loader

This class is responsible for all data handling and processing and provide an easy interface that can be used by the trainer.

--------------
### Logger

This class is responsible for the tensorboard summary, in your trainer create a dictionary of all tensorflow variables you want to summarize then pass this dictionary to logger.summarize().


This class also supports reporting to **Comet.ml** which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.
Add your API key [in the configuration file](configs/example.json#L9):

For example: "comet_api_key": "your key here"


--------------
### Comet.ml Integration

This template also supports reporting to Comet.ml which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.

Add your API key [in the configuration file](configs/example.json#L9):


For example:  `"comet_api_key": "your key here"`

Here's how it looks after you start training:
<div align="center">

<img align="center" width="800" src="https://comet-ml.nyc3.digitaloceanspaces.com/CometDemo.gif">

</div>

You can also link your Github repository to your comet.ml project for full version control.
[Here's a live page showing the example from this repo](https://www.comet.ml/gidim/tensorflow-project-template/caba580d8d1547ccaed982693a645507/chart)


--------------
### Configuration

I use Json as configuration method and then parse it, so write all configs you want then parse it using "utils/config/process_config" and pass this configuration object to all other objects.

--------------
### Main
Here's where you combine all previous part.
1. Parse the config file.
2. Create a tensorflow session.
2. Create an instance of "Model", "Data_Generator" and "Logger" and parse the config to all of them.
3. Create an instance of "Trainer" and pass all previous objects to it.
4. Now you can train your model by calling `Trainer.train()`

--------------
## Status
Project started: _09.03.2021_

## Credits
--------------
* [@mdragula](https://github.com/mdragula)
* [@MatMarkiewicz](https://github.com/MatMarkiewicz)
* [@TheFebrin](https://github.com/TheFebrin)
