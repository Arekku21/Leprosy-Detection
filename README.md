---
title: Leprosy Detection
emoji: ⚡
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 3.44.4
app_file: app.py
pinned: false
license: cc-by-4.0
---

# Leprosy Detection App

Leprosy Gradio Detection web application. In this application, I have used a fine tuned YOLOv5s model to detect Leprosy samples based on images. The application uses gradio as the platform and can also be used in the [Huggingface online hosting application](https://huggingface.co/spaces/Arekku21/Leprosy-Detection). 

## Overview
[Leprosy](https://en.wikipedia.org/wiki/Leprosy), also known as Hansen's disease, is a chronic infectious disease that primarily affects the skin and peripheral nerves. The model used is a YOLO model from [Ultralytics](https://github.com/ultralytics/yolov5) with their version YOLOv5. 

### Features

- Upload an image to the app.
- Utilizes a fine-tuned YOLOv5s model for leprosy detection.
- Detect and label leprosy regions in the uploaded image.

## Prerequisites

Before running the application, make sure you have the following prerequisites installed on your system:

- Python 3.x 
- Git
- Gradio
- Pip package manager
- Conda Virtual environment (optional but recommended)

### Installation Steps and Running the app

To install the required Python libraries, navigate to the project directory and run the following command:

#### Step 1 Clone the repository
```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/spaces/Arekku21/Leprosy-Detection

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

#### Step 2 Install requirements
```
#navigate to your cloned repository and location of requirmenents.txt
pip install -r requirements.txt
```

#### Step 3 Run the app
```
#ensure that you are using the right environment or have all the requirements installed
#ensure that you are navigated to the cloned repository
python app.py
```

#### Step 4 Using the app
Your terminal should look like this and follow the local host URL link to use the application. 
```
Downloading: "https://github.com/ultralytics/yolov5/zipball/master" to C:\Users\master.zip

YOLOv5  2023-9-28 Python-3.9.18 torch-2.0.1+cpu CPU

Fusing layers... 
Model summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
Adding AutoShape...
Running on local URL:  http://127.0.0.1:7860
```