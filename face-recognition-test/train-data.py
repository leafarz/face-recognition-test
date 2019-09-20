import cv2
import util.face_recognition_helper as frh

face_data, face_ids = frh.fetch_data('./data/images')
ids = list(face_ids.keys())

if len(ids) > 0:
    recognizer = frh.train_data(face_data, ids)
    recognizer.save('./data/train_data.yml')
else:
    print('./data/images is empty. Run create-data.py to create data.')
