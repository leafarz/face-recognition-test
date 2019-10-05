import cv2, os, json
import numpy as np
import util.face_recognition_helper as frh

# Edit only below

CONFIDENCE_THRESHOLD = 30

# Edit only above

with open('./data/training_data/list.txt', 'r') as f:
    string = f.read()

recognizers = {}
face_ids = {}
for face_part in string.split(','):
    id_file = f'./data/training_data/table_{face_part}.json'
    training_file = f'./data/training_data/train_{face_part}.yml'

    # read trained data
    if not os.path.exists(training_file):
        print(f'{training_file} doesn\'t exist!\nRun create-data.py and train-data.py to create and train data.')
        exit()

    recognizers[face_part] = cv2.face.LBPHFaceRecognizer_create()
    recognizers[face_part].read(training_file)
    test = cv2.face.LBPHFaceRecognizer_create()

    # read data table
    with open(id_file) as json_file:
        face_ids[face_part] = json.load(json_file)

# start camera
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    gray, face_data = frh.detect_face(frame)

    for face_part in face_data:
        for data in face_data[face_part]:
            (x,y,w,h) = data
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = recognizers[face_part].predict(roi_gray)
            confidence = 100 - abs(100 - confidence)
            
            if(label > len(face_ids[face_part])):
                print('face_ids mismatch with index! Training data might be outdated.')
                continue

            frh.draw_rectangle(frame, data)

            if confidence > CONFIDENCE_THRESHOLD:
                frh.draw_text(frame, face_ids[face_part][str(label)] + ',' + "{:.2f}%".format(confidence), x, y, (255,0,0))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
