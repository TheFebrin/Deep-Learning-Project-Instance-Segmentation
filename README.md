# Pet-Dataset-Instance-Segmentation
Oxford-IIIT Pet Dataset - Instance Segmentation

## Table of contents
* [General info](#general-info)
* [Libraries](#libraries)
* [Status](#status)
* [Credits](#credits)

## General info
Our goal is to create an instance segmentation model based on the Oxford-IIIT Pet Dataset.

Instance segmentation is a task in Computer Vision that aims to identify each instance of each object within the image at the pixel level. In our case, each image contains a single object, which means our goal is to predict which pixels belong to the object and which to the background as well as predict the object's class.

Our dataset contains ~200 images for each from 37 classes. Near 2/3 of those classes are dog breeds, the rest of them are cat breeds. The images have different sizes and rations, therefore they need to be resized. There's a high variance in background colors and the amount of light on the pictures.

We will build our model using [MMDetection](https://github.com/open-mmlab/mmdetection) - an open-source object detection toolbox based on PyTorch. We will experiment with different types of backbones, for example, ResNet, ResNext, or VGG.

## Libraries
* Python - version 3.7.3
* PyTorch
* numpy
* matplotlib
* seaborn

## Status
Project: WIP

## Credits
* [@mdragula](https://github.com/mdragula)
* [@MatMarkiewicz](https://github.com/MatMarkiewicz)
* [@TheFebrin](https://github.com/TheFebrin)
