import cv2 as cv
import numpy as np

from utils import stackImages, get_absolute_path
from face_detection import get_lib_face_recognition_features, lib_face_recognition, manual_face_recognition, get_manual_face_recognition_features


def main(face_recognition_method = None):
    if face_recognition_method == lib_face_recognition:
        known_encoders, known_names = get_lib_face_recognition_features('images/training/')
    else:
        known_encoders, known_names = get_manual_face_recognition_features('images/training/')

    webcam = cv.VideoCapture(0)

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        _, c_frame = webcam.read()

        c_frame = face_recognition_method(c_frame, known_encoders, known_names)

        cv.imshow('webcam', c_frame)

    webcam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(manual_face_recognition)
