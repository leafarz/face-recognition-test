import cv2
import os
import util.face_recognition_helper as frh

# Only below

DIRECTORY_NAME = "test"

# Only above

image_dir = os.path.join(os.getcwd(), "./data/images")

# create working directory if it doe
if not os.path.exists(image_dir):
    os.mkdir(image_dir)
    
working_dir = os.path.join(image_dir, DIRECTORY_NAME)
if not os.path.exists(working_dir):
    os.mkdir(working_dir)

# get starting index of filename
start_index = 0
for path, subdirnames, filenames in os.walk(working_dir):
    for filename in filenames:
        dot_index = filename.index('.')
        sub_str = filename[0:dot_index]

        try:
            index = int(sub_str)
            start_index = max(start_index, index)
        except:
            pass


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    gray, faces = frh.detect_faces(frame)

    for face in faces:
        (x,y,w,h) = face
        roi_gray = gray[y:y+h, x:x+w]

        # create files when face is detected
        filename = os.path.join(working_dir, str(start_index) + '.jpg')
        cv2.imwrite(filename, roi_gray)
        start_index += 1

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
