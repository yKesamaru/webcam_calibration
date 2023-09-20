import sys

sys.path.append('/usr/lib/python3/dist-packages')
import cv2
import numpy as np

# 保存したパラメーターの読み込み
with np.load('calib.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# 画像の読み込み
img = cv2.imread('path/to/your/image.jpg')

# 歪みの補正
dst = cv2.undistort(img, mtx, dist, None, mtx)

# 補正後の画像の表示
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)  # なにかキーを押したら終了
cv2.destroyAllWindows()
