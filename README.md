# face-recognition-test
- Test project to create images, train and recognize faces in video.

## Prerequisites
- Python 3.7

## Install (Windows)
- Install the python package:
<br>`pip install opencv-contrib-python --user`
- Create virtual environment:
<br>`python -m venv venv`
- Activate virtual environment:
<br>`venv\Scripts\activate.bat`
<br>`venv\Scripts\deactivate.bat` (to deactivate)
- Install packages from requirements:
<br>`pip install -r requirements.txt`

## Usage
1) Run `create-data.py`. This will try to recognize your face and will create images in the [DIRECTORY_NAME] in the script. Q to quit.
2) Run `train-data.py` to train the images and create the data file.
3) Run `face-recognition.py` to test.

## Tools used
- VSCode
- Python v3.7
- OpenCV