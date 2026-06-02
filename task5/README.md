
# AI Self-Driving Car Simulator

This project implements core components of an autonomous driving stack:

- Lane Detection using OpenCV
- Perspective Transform (Bird's Eye View)
- Canny + Hough Line Detection
- Basic Vehicle Steering Logic
- PID Controller Simulation
- Video Processing Support

## Features
- Detect lane lines from road images/videos
- Calculate steering direction
- Simulate vehicle control using PID
- Works with dash-cam videos
- Easy to extend with TensorFlow/PyTorch

## Folder Structure
```
ai_self_driving_car_project/
│
├── main.py
├── lane_detection.py
├── pid_controller.py
├── utils.py
├── requirements.txt
├── sample_videos/
└── output/
```

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

Press `q` to quit video playback.

## Learning Outcomes
- Computer Vision
- Autonomous Driving Basics
- Lane Detection
- Perspective Transform
- PID Control
- OpenCV Video Processing
