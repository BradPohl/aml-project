Currently only have the dataloader.

# Pose Estimation for Punching Bag Actions

This project uses a Convolutional Neural Network (CNN) along with pose estimation to classify actions related to using a punching bag. Actions include `punch`, `kick`, and `downtime` (no action). The project utilizes [MediaPipe](https://google.github.io/mediapipe/) for pose estimation and PyTorch for model training and data handling.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Usage](#usage)
4. [Dataset Preparation](#dataset-preparation)


## Project Overview
The aim of this project is to detect and classify specific actions (punches, kicks, and downtime) from video inputs using pose estimation. A CNN is trained on extracted keypoints to predict the action performed in each frame.

## Directory Structure

Your dataset should follow this structure within the project directory:

```plaintext
dataset/
├── downtime/
│   ├── downtime1.mov
│   ├── downtime2.mov
│   └── ...
├── punch/
│   ├── punch1.mov
│   ├── punch2.mov
│   └── ...
└── kick/
    ├── kick1.mov
    ├── kick2.mov
    └── ...
```

Install Dependencies:
pip install -r requirements.txt


## Usage
Run Dataloader:
python dataloader.py

This will run the dataloader and present 10 random processed videos

## Dataset Preparation
Ensure each .mov video is trimmed to contain only the specific action (punch, kick, or downtime) to improve model accuracy.
If your videos are in another format, convert them to .mov or adjust the code to support other formats.

