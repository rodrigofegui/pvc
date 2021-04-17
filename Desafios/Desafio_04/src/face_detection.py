from os import walk as walk_in
from os.path import exists

import cv2 as cv
import face_recognition
import numpy as np

from utils import get_absolute_path
from variables import CLASSIFIER_NEIGHBORS, CLASSIFIER_SCALE


def detect(img_src: np.ndarray, classifier) -> list:
    img_gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)

    return classifier.detectMultiScale(img_gray, CLASSIFIER_SCALE, CLASSIFIER_NEIGHBORS)


def get_lib_face_recognition_features(training_dir: str) -> tuple:
    """Get the known face encoders and names either based on a existence file or consulting
    the training dataset, considering JPG|JPEG|PNG images

    Args:
    - `training_dir:str`: Main training directory to look up for images

    Returns:
    - Detected encoders ans its names
    """
    known_encoders_file = get_absolute_path('../resources/known_encoders_lib_face_recognition.npy')
    known_names_file = get_absolute_path('../resources/known_names_lib_face_recognition.npy')

    known_encoders = np.load(known_encoders_file, allow_pickle=True) if exists(known_encoders_file) else np.asarray([])
    known_names = np.load(known_names_file, allow_pickle=True) if exists(known_names_file) else np.asarray([])

    if np.any(known_encoders) and any(known_names):
        return known_encoders, known_names

    known_encoders, known_names = [], []
    image_names = []
    for c_dir, _, file_names in walk_in(training_dir):
        c_img_names = [f'{c_dir}/{fn}' for fn in file_names if fn.split('.')[-1] in ['jpg', 'jpeg', 'png']]

        if not c_img_names:
            continue

        image_names.extend(c_img_names)

    for img_name in image_names:
        img = cv.cvtColor(cv.imread(img_name), cv.COLOR_BGR2RGB)

        encoded_face = face_recognition.face_encodings(img)

        if not encoded_face:
            continue

        known_encoders.append(encoded_face[0])
        known_names.append(img_name.split('/')[-2])

    known_encoders = np.asarray(known_encoders)
    known_names = np.asarray(known_names)

    np.save(known_encoders_file, known_encoders)
    np.save(known_names_file, known_names)

    return known_encoders, known_names


def lib_face_recognition(base_image: np.ndarray, known_encoders: np.array, known_names: np.array) -> np.ndarray:
    """Face recognition using the homonymous module, which implements the HOG
    (Histogram of Oriented Gradient). The detected faces are marked within a square
    and labeled as `face_name` or `'unknown'`

    Args:
    - `base_image:np.ndarray`: Base image to face recognition
    - `known_encoders:np.array`: Known face encoders to recognition match search
    - `known_names:np.array`: Known face names to recognition match search

    Returns:
    - Marked image
    """
    face_locations = face_recognition.face_locations(base_image)
    c_encoded_faces = face_recognition.face_encodings(base_image, face_locations)

    for ind, encoded_face in enumerate(c_encoded_faces):
        comparisons = face_recognition.compare_faces(known_encoders, encoded_face)
        face_dist = face_recognition.face_distance(known_encoders, encoded_face)
        min_dist_at = np.argmin(face_dist)

        y1, x2, y2, x1 = face_locations[ind]
        cv.rectangle(base_image, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if comparisons[min_dist_at]:
            face_name = f'{known_names[min_dist_at]}: {face_dist[min_dist_at]}'
        else:
            face_name = f'unknown: {face_dist[min_dist_at]}'

        cv.putText(base_image, face_name, (x1, y1), cv.FONT_HERSHEY_COMPLEX, .7, (115, 249, 255), 1, cv.LINE_AA)

    return base_image
