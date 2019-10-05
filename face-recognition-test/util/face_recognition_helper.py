import cv2, math, os
import numpy as np

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    eye_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray)

    return gray, { "faces": faces, "eyes": eyes}

def draw_rectangle(img, face):
    channels = (255,) * img.shape[2] if len(img.shape) > 2 else 1
    x,y,w,h = face
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)

def draw_text(img, text, x, y, color):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def fetch_data(root_directory):
    curr_id = {}
    ret_face_data = {}
    ret_face_ids = {}
    
    for path, subdirnames, filenames in os.walk(root_directory):
        if path == root_directory: continue
        if len(subdirnames) > 0:
            user = os.path.basename(path)
            continue
        # face_part = os.path.basename(path)

        for filename in filenames:
            print(f'Reading: {filename}')

            if filename.startswith('.'): continue
            img_path = os.path.join(path, filename)
            
            img = cv2.imread(img_path)
            if img is None: continue

            gray, face_data = detect_face(img)
            for face_part in face_data:
                data = face_data[face_part]
                if len(data) != 1: continue

                if face_part not in ret_face_ids: ret_face_ids[face_part] = {}
                if face_part not in ret_face_data: ret_face_data[face_part] = []
                if face_part not in curr_id: curr_id[face_part] = 0

                ret_face_ids[face_part][curr_id[face_part]] = user
                curr_id[face_part] += 1

                (x,y,w,h) = data[0]
                roi_gray = gray[y:y+h, x:x+w]

                ret_face_data[face_part].append(roi_gray)
    return ret_face_data, ret_face_ids
                

def train_data(face_data, face_id):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_data, np.array(face_id))
    return recognizer

def resize_by_area(img, max_area):
    h,w,_ = img.shape
    dim = w * h

    # if dimension is greater than max_area, we resize it to that area
    if dim > max_area:
        val = max_area / dim
        k = math.sqrt(val)
        w = int(w * k)
        h = int(h * k)
        img = cv2.resize(img, (w,h))

    return img