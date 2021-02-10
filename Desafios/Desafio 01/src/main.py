import cv2
import numpy as np
from utils import stackImages, get_background_image
from virtual_background import background_subtraction, face_detection


def main(processing_method):
    webcam = cv2.VideoCapture(0)

    background_img = get_background_image(webcam)

    _, fst_frame = webcam.read()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        _, cur_frame = webcam.read()

        mask = processing_method(cur_frame, fst_frame)

        masked_frame = cv2.bitwise_and(cur_frame, cur_frame, mask=mask)
        indexes = np.where(masked_frame == 0)
        masked_frame[indexes] = background_img[indexes]

        stacked_img = stackImages(.5, ([cur_frame, background_img, masked_frame]))
        cv2.imshow("Imagens", stacked_img)

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(face_detection)
