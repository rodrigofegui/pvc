import cv2 as cv
from src.camera_handler import get_camera_matrices, get_real_and_img_pts
from src.utils import draw_cursor, draw_clicks, calculate

CHESS_GRID_SHAPE = (7, 7)
REAL_CHESS_SQ_SIZE = 2.1
IMG_CHESS_SRC_PATTERN = './images/*.jpg'
SHOW_CHESS_ANALYSES = False
WEBCAM_WIN_NAME = 'Webcam'
INPUT_WIN_NAME = 'Known data'
FONT = cv.FONT_HERSHEY_COMPLEX

real_pts, img_pts, img_shape, img_names = get_real_and_img_pts(
    CHESS_GRID_SHAPE, IMG_CHESS_SRC_PATTERN, REAL_CHESS_SQ_SIZE, SHOW_CHESS_ANALYSES
)
matrices = get_camera_matrices(real_pts, img_pts, img_shape)
obj_pts = []
cursor_lines = []

def mouse_handler(event, x, y, flags, param):
    global cur_frame, obj_pts, cursor_lines

    if event == cv.EVENT_LBUTTONDOWN:
        obj_pts.append((x, y))
    elif event == cv.EVENT_MOUSEMOVE:
        cursor_lines = [
            ((x, 0), (x, cur_frame.shape[0])),
            ((0, y), (cur_frame.shape[1], y))
        ]

def distance_handler(a):
    print('distance_handler', a)
    cv.setTrackbarPos('Size (cm)', INPUT_WIN_NAME, 0)

def size_handler(s):
    print('size_handler', s)
    cv.setTrackbarPos('Distance (cm)', INPUT_WIN_NAME, 0)

webcam = cv.VideoCapture(0)
cv.namedWindow(WEBCAM_WIN_NAME)
cv.setMouseCallback(WEBCAM_WIN_NAME, mouse_handler)

cv.namedWindow(INPUT_WIN_NAME)

cv.createTrackbar('Distance (cm)', INPUT_WIN_NAME, 0, 200, distance_handler)
cv.createTrackbar('Size (cm)', INPUT_WIN_NAME, 0, 100, size_handler)

resp = ''
while True:
    pressed_key = cv.waitKey(1) & 0xFF

    if pressed_key == ord('q'):
        break
    elif pressed_key == ord('r'):
        obj_pts.clear()
        resp = ''
    elif pressed_key == ord('\r'):
        resp = calculate(
            matrices,
            obj_pts,
            cv.getTrackbarPos('Distance (cm)', INPUT_WIN_NAME),
            cv.getTrackbarPos('Size (cm)', INPUT_WIN_NAME)
        )

    _, cur_frame = webcam.read()

    cur_frame = draw_cursor(cur_frame, cursor_lines)
    cur_frame = draw_clicks(cur_frame, obj_pts)

    cv.rectangle(cur_frame, (0, 0), (cur_frame.shape[1], 30), (38, 0, 77), cv.FILLED)
    cv.putText(cur_frame, 'q: quit | r: reset points | ENTER: calculate', (70, 20), FONT, .7, (255, 249, 115), 1, cv.LINE_AA)

    if resp:
        cv.putText(cur_frame, resp, (5, cur_frame.shape[0] - 20), FONT, .7, (255, 249, 115), 1, cv.LINE_AA)

    cv.imshow(WEBCAM_WIN_NAME, cur_frame)

webcam.release()
cv.destroyAllWindows()
