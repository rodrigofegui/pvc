import cv2 as cv
from datetime import datetime

from utils.utils import get_disp_map, get_resize_shape, parse_calib_file
from utils.variables import WORKDIRS, DISP_BLOCK_SEARCH, RESULT_DIR

print('Challenge 1: Disparities and deepness\n')

for workdir in WORKDIRS:
    calib = parse_calib_file(f'{workdir}/calib.txt')
    print(f'calib: {calib}\n')
    height, width = get_resize_shape(calib['shape'])

    img_left = cv.imread(f'{workdir}/im0.png', cv.IMREAD_UNCHANGED)
    img_left = cv.resize(img_left, (height, width), interpolation=cv.INTER_AREA)

    img_right = cv.imread(f'{workdir}/im1.png', cv.IMREAD_UNCHANGED)
    img_right = cv.resize(img_right, (height, width), interpolation=cv.INTER_AREA)
    print('img_left', img_left.shape)
    print('img_right', img_right.shape)
    print()

    # disp_map = get_disp_map(img_left, img_right, use_default=True)
    # cv.imwrite(f'{RESULT_DIR}/disp_{datetime.now().isoformat()}.png', disp_map)
    disp_map = get_disp_map(img_left, img_right)
    cv.imwrite(f'{RESULT_DIR}/disp_{datetime.now().isoformat()}.png', disp_map)

print('\nEnding: Challenge 1')
print('-' * 50)
cv.destroyAllWindows()
