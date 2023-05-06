# LDAR_Detection
## **1.Goals**

The goals of this project are:

-Our project is to detect whether the distance from the probe to the pipeline meets the criteria, utilizing machine learning

-This project aims to perform object detection and distance measurement using YOLOv5 object detection model.

## **2.How to work**

### Install Python & Conda

Before you start, you will need [Python](https://wiki.python.org/moin/BeginnersGuide/Download) and [Conda](https://docs.anaconda.com/anaconda/install/) on your computer.

### **2.1 YOLO**

#### 2.1.1  Prerequisites

  Python 3.6 or higher

  PyTorch 1.7.0 or higher

  OpenCV 4.2.0 or higher

  NumPy 1.18.0 or higher
  
#### 2.1.2  Installation

  1.Clone this repository(use http/ssh):

  ```bash
 git clone https://github.com/AlexandreQ27/LDAR_Detection.git
  ```
  
> **Warning**
>
> You may see below errors that prevent you from connecting to the remote repository, or timeout errors when you do push operations, especially if you are using the HTTP protocol.
>
> ```bash
> Permission denied (publickey).
> fatal: Could not read from remote repository.
> fatal: unable to access 'https://github.com/AlexandreQ27/LDAR_Detection.git': Recv failure: Connection was reset.
> fatal: unable to access 'https://github.com/AlexandreQ27/LDAR_Detection.git': The requested URL returned error : 403.
> ```
>
> Solution:
>
> 1. Use [SSH protocol](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) to access the repo.
> 2. Try more times in case the push operation fails occasionally.
  
  2.Install  YOLOv5
  
  ```bash
  git clone https://github.com/ultralytics/yolov5.git
  ```
  
  3.Install the required packages:

  ```bash
  cd yolov5
  pip install -r requirements.txt
  ```
  
  4.Download the YOLOv5 model weights from the official repository or from my project(yolov5s.pt)

#### 2.1.3  Acknowledgments

  -YOLOv5: https://github.com/ultralytics/yolov5

  -OpenCV: https://opencv.org/

  -PyTorch: https://pytorch.org/
  
  ### **2.1 Monodepth2**

#### 2.1.1  Prerequisites
  certifi==2021.5.30
  cycler==0.10.0
  decorator==4.4.2
  imageio==2.9.0
  kiwisolver==1.3.1
  matplotlib==3.3.4
  networkx==2.5.1
  numpy==1.19.5
  Pillow==8.3.1
  pyparsing==2.4.7
  python-dateutil==2.8.2
  PyWavelets==1.1.1
  scikit-image==0.17.2
  scipy==1.5.4
  six==1.16.0
  tifffile==2020.9.3
  torch==0.4.1
  torchvision==0.4.1+cpu
  wincertstore==0.2


  
