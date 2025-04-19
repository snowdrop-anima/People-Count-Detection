# People Detection and Logging System
This Python script uses a pre-trained YOLOv5 model to detect people from a webcam stream in real-time. It logs when each person is first seen and last seen based on a simple bounding box center tracking method. At the end of the session, it saves the detection history in a CSV file.

## Features
##### Real-time person detection using YOLOv5 (yolov5s model).
##### Automatic webcam detection (tries up to 5 devices).
##### Tracks appearance history of each person based on bounding box center.
##### Saves the history (ID, First Seen, Last Seen) to people_history.csv.
##### Visualization of bounding boxes and person IDs on the video stream.

## Requirements
##### Python 3.7+
##### pytorch
##### opencv-python
##### pandas
