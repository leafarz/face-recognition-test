import cv2, json, os
from util import face_recognition_helper as frh

ROOT_DIR = './data/face_data'

face_data, face_ids = frh.fetch_data(ROOT_DIR)

with open('./data/training_data/list.txt', 'w') as f:
    f.write(','.join(face_ids.keys()))

for face_part in face_ids:
    ids = list(face_ids[face_part].keys())
    if len(ids) > 0:
        recognizer = frh.train_data(face_data[face_part], ids)
        recognizer.save(f'./data/training_data/train_{face_part}.yml')

        with open(f'./data/training_data/table_{face_part}.json', 'w') as outfile:
            json.dump(face_ids[face_part], outfile)
    else:
        print('Data is empty. Run extract-images.py to create data.')
