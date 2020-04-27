import json
import os

import cv2

from util import face_recognition_helper as frh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA = os.path.join(BASE_DIR, "../data/training_data")
FADE_DATA_DIR = os.path.join(BASE_DIR, "../data/face_data")

if not os.path.exists(TRAINING_DATA):
    os.makedirs(TRAINING_DATA)

face_data, face_ids = frh.fetch_data(FADE_DATA_DIR)

with open(os.path.join(TRAINING_DATA, "list.txt"), "w") as f:
    f.write(",".join(face_ids.keys()))

for face_part in face_ids:
    ids = list(face_ids[face_part].keys())
    if len(ids) > 0:
        recognizer = frh.train_data(face_data[face_part], ids)
        recognizer.save(os.path.join(TRAINING_DATA, f"train_{face_part}.yml"))

        with open(
            os.path.join(TRAINING_DATA, f"table_{face_part}.json"), "w"
        ) as outfile:
            json.dump(face_ids[face_part], outfile)
    else:
        print("Data is empty. Run extract-images.py to create data.")
