from src.camera_handler import get_camera_matrices, get_real_and_img_pts

CHESS_GRID_SHAPE = (7, 7)
REAL_CHESS_SQ_SIZE = 2.1
IMG_CHESS_SRC_PATTERN = './images/*.jpg'
SHOW_CHESS_ANALYSES = False

real_pts, img_pts, img_shape = get_real_and_img_pts(CHESS_GRID_SHAPE, IMG_CHESS_SRC_PATTERN, REAL_CHESS_SQ_SIZE, SHOW_CHESS_ANALYSES)

intrinsics, distortion = get_camera_matrices(real_pts, img_pts, img_shape)

print('Intrinsic Matrix (K):')
print(intrinsics, end='\n\n')

print('Distortion Coefficients:')
print(distortion, end='\n\n')
