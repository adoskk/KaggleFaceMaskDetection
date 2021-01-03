# KaggleFaceMaskDetection

## Introduction

This is a PyTorch project using Faster RCNN for 2-class face mask detection.

For Faster RCNN tutorial, please see: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

## Dataset Description

Kaggle face mask detection dataset: https://www.kaggle.com/andrewmvd/face-mask-detection

- contains 853 images
- each image is accompanied by an annotation file, including multiple bounding boxes and labels
- 3-classes annotation is available: with_mask, without_mask, mask_weared_incorrect (not used in this project)

## Folder Structure

FaceMaskDetection

|---- data  
|     |---- original_data

|      |      |---- images

|      |      |---- annotations  

|---- utilities

|      |---- coco_utils

|      |---- data_utils

|      |---- train_eval

|---- output

|---- model

README.md

requirements.txt

setup.py

train.py

test.py

## Environment Setup

The project was written under the following environment:

- Python==3.7
- torch==1.4.0
- torchvision==0.5.0
- pycocotools==2.0.2
- nvidia-ml-py3
- xml
- PIL

Before running any other code, please **run the setup.py in the root folder first**, so to setup the file paths properly.

## Data Preprocessing

1. Download the dataset and put the images and annotation files under the corresponding folders
2. go to utilities--> data_utils--> split_dataset.py
3. now check the data folder, there should be two new sub-folders now: train and test

## Train and Test

- For training, please run train.py so to generate the model; or you can simply download the pre-trained model from: https://drive.google.com/drive/folders/1UuU5up0DQfRuafdbZrayPdRwpVvJOYYK?usp=sharing
- the testing results will be written into the output folder, here's an example of prediction:![Example Output](https://github.com/adoskk/KaggleFaceMaskDetection/blob/master/output/result4.png)
