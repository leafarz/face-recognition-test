import cv2
import util.face_recognition_helper as frh
import json


face_data, face_ids = frh.fetch_data('./data/faces')
ids = list(face_ids.keys())

if len(ids) > 0:
    recognizer = frh.train_data(face_data, ids)
    recognizer.save('./data/train_data.yml')

    with open('./data/data_table.json', 'w') as outfile:
        json.dump(face_ids, outfile)
else:
    print('./data/images is empty. Run create-data.py to create data.')
