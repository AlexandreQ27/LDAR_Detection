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

1. Clone this repository(use http/ssh):

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

2. Install  YOLOv5

  ```bash
  git clone https://github.com/ultralytics/yolov5.git
  ```

3. Install the required packages:

  ```bash
  cd yolov5
  pip install -r requirements.txt
  ```

4. Download the YOLOv5 model weights from the official repository or from my project(yolov5s.pt)

#### 2.1.3  Acknowledgments

  -YOLOv5: https://github.com/ultralytics/yolov5

  -OpenCV: https://opencv.org/

  -PyTorch: https://pytorch.org/

  ### **2.2 Monodepth2**

#### 2.2.1  Prerequisites

Python 3.6

PyTorch 0.4.1

OpenCV 4.1

#### 2.2.2 Installation

1. Install Monodepth2

If this is your first time cloning this repository, use the following command will install both YOLOv5 and Monodepth2.

```bash
git submodule update --init --recursive
```

2. Install the required packages:

The requirements.txt is in our repository's directory: `requirements`, create a new Python virtual environment for monodepth2 and use the following command.

```bash
pip install -r .\requirements.txt
```

> **Warning**
>
> You may see errors below that prevent you from installing PyTorch.
>
> ```bash
> ERROR: Could not find a version that satisfies the requirement torch==0.4.1 (from versions: 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2)
> ERROR: No matching distribution found for torch==0.4.1
> ```
>
> Solution:
>
> ```
> pip install torch==0.4.1 -f https://download.pytorch.org/whl/torch_stable.html-f https://download.pytorch.org/whl/torch_stable.html

#### 2.2.3  Acknowledgments

  -Monodepth2: https://github.com/nianticlabs/monodepth2

  
