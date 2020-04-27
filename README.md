# face-recognition-test

- Test project to create images, train and recognize faces in video using Haar Cascade face and eye classifiers.

## Prerequisites

- Python 3.7

## Install (Windows)

1. Create virtual environment:
   ```
   python -m venv venv
   ```
2. Activate virtual environment:
   ```
   venv\Scripts\activate
   venv\Scripts\deactivate :: to deactivate
   ```
3. Install packages from requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

- From the inner `face-recognition-test` directory
- Creating image data can be done in 2 ways:

  1. Create image data via recording

     - Run command
       ```
       python create-data.py
       ```
     - This will try to capture your face and eyes from camera and will create images in the `./data/face_data/[DIRECTORY_NAME]` directory.
     - DIRECTORY_NAME is found in the script.
     - [R] to record and [Q] to quit.

  2. Extract image data from existing images
     - Run command
       ```
       python extract-images.py
       ```
     - This will get faces from images in `./data/images/` and extract it to `./data/face_data` directory. Refer to [Directories section](#sec_dir).

- Train the image data

  - Run command to train the images in the faces folder and create the trained data file
    ```
    python train-data.py
    ```

- Run the face recognition program
  ```
  python face-recognition.py
  ```
  - [Q] to quit.

<a name="sec_dir"></a>

## Directories

- Root directory of images `./data/images/[name_of_person]` (must be inside the folder with the identifier's name)
- Root directory of created or extracted images to train: `./data/images/face_data`

## Tools used

- VSCode
- Python v3.7
- OpenCV
