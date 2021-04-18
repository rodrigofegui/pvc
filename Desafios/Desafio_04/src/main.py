import cv2 as cv
from data_aquisition import DataAquisition
from face_recognizer import FaceRecognizer


def main(face_recognition_method: str = None):
    webcam = cv.VideoCapture(0)

    data_aquisition = DataAquisition(
        dataset_dir='images/training/',
        resources_dir='resources/'
    )

    known_people = data_aquisition.create_dataset(webcam)

    if len(known_people) == 1:
        print('Should have any face to recognition')
        return

    print('Training model...')
    data_aquisition.train(face_recognition_method)

    print('Initialzing face recognizer...')
    face_recognizer = FaceRecognizer(
        resources_dir='resources/',
        recognizer_name=face_recognition_method
    )

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        _, c_frame = webcam.read()

        c_frame, face_locations, labels, factors = face_recognizer.predict(c_frame, known_people)

        cv.imshow('webcam', c_frame)

    webcam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    """Available methods: 'lib_face_recognition' | 'heuristic' | 'lbph_machinelearning'
    """
    # main('lib_face_recognition')
    # main('heuristic')
    main('lbph_machinelearning')
