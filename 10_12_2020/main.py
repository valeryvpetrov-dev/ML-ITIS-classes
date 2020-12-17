import sys

import cv2
from datetime import datetime


def take_camera_picture():
    camera = cv2.VideoCapture(0)
    input('Press Enter to capture')
    return_value, image = camera.read()
    del (camera)
    return image


def save_result(image):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.imwrite(time + '.png', image)


def detect_faces(img, img_gray):
    face_cascade = cv2.CascadeClassifier(
        '../venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        '../venv/lib/python3.8/site-packages/cv2/data/haarcascade_eye.xml'
    )

    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('img', img)


if __name__ == '__main__':
    img = take_camera_picture()
    # img = cv2.imread('test.jpeg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect_faces(img, img_gray)
    if cv2.waitKey(0) == 27:  # close on ESC key
        save_result(img)
        cv2.destroyAllWindows()
        sys.exit()
