import os

import cv2

from util import face_recognition_helper as frh

# Edit below only
DIRECTORY_NAME = "Name"
# Edit above only


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "../data/images")
DATA_DIR = os.path.join(BASE_DIR, "../data/face_data")

# create directories if it not found
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

user_dir = os.path.join(DATA_DIR, DIRECTORY_NAME)
if not os.path.exists(user_dir):
    os.mkdir(user_dir)

# get starting index of filename
start_indices = {}
for path, subdirnames, filenames in os.walk(user_dir):
    if path == user_dir:
        continue

    key = os.path.basename(path)
    if key not in start_indices:
        start_indices[key] = 0

    for filename in filenames:
        dot_index = filename.index(".")
        sub_str = filename[0:dot_index]

        try:
            index = int(sub_str)
            start_indices[key] = max(start_indices[key], index)
        except:
            pass

record = False
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    gray, data = frh.detect_face(frame)

    if record:
        frh.draw_text(frame, "Recording", 0, 30, (0, 0, 255))
        for face_part in data:
            base_path = os.path.join(user_dir, face_part)

            if not os.path.exists(base_path):
                os.mkdir(base_path)

            if face_part not in start_indices:
                start_indices[face_part] = 0

            for d in data[face_part]:
                (x, y, w, h) = d
                roi_gray = gray[y : y + h, x : x + w]

                # create files when face is detected
                filename = os.path.join(
                    base_path, str(start_indices[face_part]) + ".jpg"
                )

                print(f"Creating to: {filename}")
                cv2.imwrite(filename, roi_gray)
                start_indices[face_part] += 1
    else:
        frh.draw_text(frame, "Not recording", 0, 30, (0, 255, 0))

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("r"):
        record = not record


cap.release()
cv2.destroyAllWindows()
