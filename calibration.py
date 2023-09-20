import sys

sys.path.append('/usr/lib/python3/dist-packages')
import glob

import cv2
import numpy as np

# チェスボードの内角の数（マス目の数は7x10）
CHECKERBOARD = (6,9)  # (列,行)。印刷したチェスボードに合わせて変更してください。

# 3Dのワールド座標系の点
objpoints = []
for i in range(CHECKERBOARD[1]):
    for j in range(CHECKERBOARD[0]):
        objpoints.append([j, i, 0])

objpoints = np.array(objpoints, dtype=np.float32)

# 2Dと3Dの点を保存するリスト
objpoints_list = [] # 3Dのワールド座標系の点
imgpoints_list = [] # 2Dの画像上の点

# 画像の読み込み
images = glob.glob('path/to/your/images/*.jpg') # 画像のパスを指定

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print(f"チェスボードの検出に失敗しました: {fname}")
        objpoints_list.append(objpoints)
        imgpoints_list.append(corners)

# キャリブレーション
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_list, imgpoints_list, gray.shape[::-1], None, None)

# 結果の確認
img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# パラメータの保存
np.savez('calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("キャリブレーションが完了しました。")
