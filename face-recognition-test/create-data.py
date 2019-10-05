import cv2, os
from util import face_recognition_helper as frh

# Edit below only

DIRECTORY_NAME = "raf"

# Edit above only

img_dir = os.path.join(os.getcwd(), "./data/images")
data_dir = os.path.join(os.getcwd(), "./data/face_data")

# create directories if it not found
if not os.path.exists(img_dir): os.mkdir(img_dir)
if not os.path.exists(data_dir): os.mkdir(data_dir)

user_dir = os.path.join(data_dir, DIRECTORY_NAME)
if not os.path.exists(user_dir):
    os.mkdir(user_dir)

# get starting index of filename
start_indices = {}
for path, subdirnames, filenames in os.walk(user_dir):
    if path == user_dir: continue

    key = os.path.basename(path)
    if key not in start_indices: start_indices[key] = 0

    for filename in filenames:
        dot_index = filename.index('.')
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
        frh.draw_text(frame, "Recording", 0, 30, (0,0,255))
        for face_part in data:
            base_path = os.path.join(user_dir, face_part)

            if not os.path.exists(base_path):
                os.mkdir(base_path)
            
            if face_part not in start_indices: start_indices[face_part] = 0
            
            for d in data[face_part]:
                (x,y,w,h) = d
                roi_gray = gray[y:y+h, x:x+w]

                # create files when face is detected
                filename = os.path.join(base_path, str(start_indices[face_part]) + '.jpg')

                print(f'Creating to: {filename}')
                cv2.imwrite(filename, roi_gray)
                start_indices[face_part] += 1
    else:
        frh.draw_text(frame, "Not recording", 0, 30, (0,255,0))

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        record = not record


cap.release()
cv2.destroyAllWindows()
