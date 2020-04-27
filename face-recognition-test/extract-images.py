import os

import cv2

from util import face_recognition_helper as frh

# Edit only below
THRESHOLD_AREA = 1024 * 1024
# Edit only above


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "../data/images")
FACE_DATA_DIR = os.path.join(BASE_DIR, "../data/face_data")

if not os.path.exists(IMG_DIR):
    print(f"{IMG_DIR} not found.\nNo images to extract.")
    exit()

# create directories if it not found
if not os.path.exists(FACE_DATA_DIR):
    os.mkdir(FACE_DATA_DIR)

for path, subdirnames, filenames in os.walk(IMG_DIR):
    user = os.path.basename(path)

    for filename in filenames:
        print(f"Reading: {filename}")
        img = cv2.imread(os.path.join(path, filename))
        if img is None:
            continue

        # do I mess up face detection quality here?
        img = frh.resize_by_area(img, THRESHOLD_AREA)

        gray, face_data = frh.detect_face(img)
        for face_part in face_data:
            if len(face_data[face_part]) != 1:
                continue

            for data in face_data[face_part]:
                x, y, w, h = data
                roi_gray = gray[y : y + h, x : x + w]

                face_dir = os.path.join(FACE_DATA_DIR, user)
                if not os.path.exists(face_dir):
                    os.mkdir(face_dir)

                face_dir = os.path.join(face_dir, face_part)
                if not os.path.exists(face_dir):
                    os.mkdir(face_dir)

                out_file = os.path.join(face_dir, filename)
                print(f"Extracting [{face_part}] data to: {out_file}")
                cv2.imwrite(out_file, roi_gray)

print("DONE")
