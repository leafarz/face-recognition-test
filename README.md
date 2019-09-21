# face-recognition-test
- Test project to create images, train and recognize faces in video.

## Prerequisites
- Python 3.7

## Install (Windows)
- Create virtual environment:
<br>`python -m venv venv`
- Activate virtual environment:
<br>`venv\Scripts\activate.bat`
<br>`venv\Scripts\deactivate.bat` (to deactivate)
- Install packages from requirements:
<br>`pip install -r requirements.txt`

## Usage
- From root, run python scripts via `python [path_to_script]` command
- Creating Images
  - Run `create-data.py`
  - This will try to recognize your face from camera and will create images in the `./data/faces/[DIRECTORY_NAME]` directory.
  - DIRECTORY_NAME is found in the script.
  - Q to quit.
- Extracting Existing Images
  - Run `extract-images.py`
  - This will get faces from images in `./data/images/` and extract it to `./data/faces` directory. Refer to [Directories section](#sec_dir).
- Train the Data
  - Run `train-data.py` to train the images in the faces folder and create the trained data file.
- Run the face recognition program
  - Run `face-recognition.py` to test.
  - Q to quit.

<a name="sec_dir"></a>
## Directories
- Root directory of images `./data/images/[name_of_person]` (must be inside the folder with the identifier's name)
- Root directory of created or extracted images to train: `./data/images/faces`

## Tools used
- VSCode
- Python v3.7
- OpenCV