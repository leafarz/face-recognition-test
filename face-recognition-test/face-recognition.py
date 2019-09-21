import cv2
import os
import json
import numpy as np
import util.face_recognition_helper as frh

# Edit only below

CONFIDENCE_THRESHOLD = 30

# Edit only above

# read trained data
if not os.path.exists('./data/train_data.yml'):
    print('train_data.yml doesn\'t exist!\nRun create-data.py and train-data.py to create and train data.')
    exit()

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('./data/train_data.yml')

# read data table
with open('./data/data_table.json') as json_file:
    face_ids = json.load(json_file)

# start camera
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    gray, faces = frh.detect_faces(frame)

    # add eyes detection?
    for face in faces:
        (x,y,w,h) = face
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi_gray)
        
        if(label > len(face_ids)):
            print('face_ids mismatch with index! Training data might be outdated.')
            continue

        frh.draw_rectangle(frame, face)

        if confidence > CONFIDENCE_THRESHOLD:
            frh.draw_text(frame, face_ids[str(label)] + ',' + "{:.2f}%".format(confidence), x, y, (255,0,0))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
