# Traffic Counting Program using YOLOv8

This software project is designed to automate traffic counting using YOLOv8, a state-of-the-art object detection model. It can be used to count vehicles in a video stream or a camera feed. The program is written in Python 3.10 and utilizes the Ultralytics package, which includes YOLOv8. This README provides an overview of the project, installation instructions, and usage details.

## Table of Contents

- [Introduction](#traffic-counting-program-using-yolov8)
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [YOLOv8 Weights](#yolov8-weights)
- [Output](#output)
- [Python Version](#python-version)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Overview

The Traffic Counting Software using YOLOv8 is a versatile tool for traffic analysis. Key features and components include:

- Object Detection: YOLOv8 is used for real-time vehicle detection in video streams.
- Video Sources: You can use either a video file or a connected camera as the input source.
- File Handling: The program handles video files and saves traffic count information to a file.
- YOLO Weights: Pre-trained YOLOv8 weights are provided in the "weights" folder.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- Microsoft Windows® 11 as the operating system (OS).
- Python 3.10 as the development environment.
- Required Python packages (you can install them using `pip install -r requirements.txt`).
- Ultralytics package, which contains YOLOv8.
- Access to a video source (file path or attached camera).

### Installation

1. Clone this repository to your local machine:

   ```shell
   git clone https://github.com/anujeshify/Traffic-Counting-Program-using-YOLOv8.git
   ```

2. Change your current directory to the project folder:

   ```shell
   cd Traffic-Counting-Program-using-YOLOv8
   ```

3. Install the required packages using pip:

   ```shell
   pip install -r requirements.txt
   ```

4. Ensure you have the YOLOv8 weights file (`yolov8m.pt`) in the `Yolo-Weights` folder.

## Usage

1. Run the `TrafficCounter.py` script in PyCharm or your preferred Python IDE.

2. Modify the video source by changing the file path or using a connected camera. You can do this in the script.

3. The program will detect and count vehicles in the video stream.

4. The vehicle count information will be saved in `vehicle_count.txt`.

## YOLOv8 Weights

This project includes pre-trained YOLOv8 weights (`yolov8m.pt`) located in the `Yolo-Weights` folder. You can use this file for object detection. You can also experiment with other YOLOv8 weights like `yolov8l.pt` and `yolov8n.pt` for different model variants.

## Output

The program will save the vehicle count in the `vehicle_count.txt` file in the project directory.
![Traffic Counting](https://github.com/anujeshify/Traffic-Counting-Program-using-YOLOv8/blob/main/output.png)

here: vid - vehicle id


## Built With

* [Python](https://www.python.org/doc/) - Front-End Application
* [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 Object Detection
* [OpenCV](https://opencv.org/) - Computer Vision Library
* [NumPy](https://numpy.org/doc/) - Numerical Computing Library

## Authors

* **Anujesh Bansal** - *Initial work* - [your_username](https://github.com/your_username)

## Acknowledgments

* Inspiration - This project was inspired by the need for accurate traffic analysis and management using concepts of deep learning and neural networks.
