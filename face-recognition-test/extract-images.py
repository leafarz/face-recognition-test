import cv2
import os
import math
import util.face_recognition_helper as frh

# Edit only below
THRESHOLD_AREA = 1024*1024
# Edit only above

img_dir = os.path.join(os.getcwd(), "./data/images")
out_dir = os.path.join(os.getcwd(), "./data/faces")

if not os.path.exists(img_dir):
    print(f'{img_dir} not found.\nNo images to extract.')
    exit()

# create directories if it not found
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for path, subdirnames, filenames in os.walk(img_dir):
    id = os.path.basename(path)

    for filename in filenames:
        print(f'Reading: {filename}')
        img = cv2.imread(os.path.join(path,filename))
        if img is None: continue

        # do I mess up face detection quality here?
        img = frh.resize_by_area(img, THRESHOLD_AREA)
        
        gray, faces = frh.detect_faces(img)
        if len(faces) != 1: continue

        for face in faces:
            x,y,w,h = face
            roi_gray = gray[y:y+h, x:x+w]
            
            face_dir = os.path.join(out_dir, id)
            if not os.path.exists(face_dir):
                os.mkdir(face_dir)
            
            out_file = os.path.join(face_dir, filename)
            print(f'Extracting to: {out_file}')
            cv2.imwrite(out_file, roi_gray)
