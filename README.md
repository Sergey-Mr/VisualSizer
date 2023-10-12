# Body-measurement python library

* [General info](#general-info)
* [Technologies](#technologies)
* [Repository explanation](#reposityory-explanation)
* [Calculation logic](#calculation-logic)
* [Development plan](#development-plan)

## General Info
This python library is designed to provide accurate measurements of different parts of the human body from images of people. It is aimed to increase shopping experience foor the users of online clothing shops and reduce return rates for retailers.

## Technologies
Project is created with:
* Python 3.10.12
* OpenCV 4.6.0-dev
* OpenPose v1.7.0
* Cuda 11.7.r11.7
* PyTorch 1.13.1+cu117
* YOLO v5

## Repository explanation
4 main code are used in the body measurement process
1) main_working_2.py to find length of hands, shoulders, run openpose, analyze points of body
2) roi-find.py uses yolo to find a human and restrict ROI
3) jsond_data_out.py to get points after openpose and put them into main_working_2.py
4) pytorch_working_01.py to get a mask of the body (white-black) 

## Calculation logic
Explain the segmentation idea, refer to particular reserches, explain how the arms and shoulders were calculated

## Development plan
Summary of the development plan
