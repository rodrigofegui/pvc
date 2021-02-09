import cv2
import numpy as np
from utils import get_background_image
from virtual_background import background_subtraction


def main():
    webcam = cv2.VideoCapture(0)

    background_img = get_background_image(webcam)

    background_subtraction(webcam, background_img)

    webcam.release()
    cv2.destroyAllWindows()

def empty(val):
    pass

if __name__ == '__main__':
    main()
