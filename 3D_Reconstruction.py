from typing import ChainMap
import numpy as np
import cv2
import glob
from numpy.core.shape_base import block, stack
from numpy.lib.function_base import disp
import open3d as o3d

supervise_mode = True

# Your information here

if supervise_mode:

# ====================================================================================
# Camera calibration
# ====================================================================================
# Set directory path (images capturing check pattern)
# Example) calibration_dir_path = 'calibration/*.png'
calibration_dir_path =
calibration_images = glob.glob(calibration_dir_path)

# intrinsic parameters and distortion coefficient
# With these parameter, you can get undistorted image and new intrinsic parameter of them (K_undist)
K = np.array([], dtype=np.float32)
dist = np.array([], dtype=np.float32)
# new matrix for undistorted intrinsic parameter
K_undist = np.array([], dtype=np.float32)

# reference: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.htmls

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 21, 0.001)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

shape = (1, 1)

for fname1 in calibration_images:
    img = cv2.imread(fname1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    shape = gray.shape[::-1]
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

    ret, K, dist, rvecs, tvexs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    img = cv2.imread ('calibration_images_folder/1.jpg')
    h, w = img.shape[:2]
    K_undist, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

if supervise_mode:
    print('1-1. Calibration: K matrix')
    print(K)
    print('1-2. Calibration: distortion coefficients')
    print(dist)
    print('1-3. Calibration: Undistorted K matrix')
    print(K_undist)
    for fname in calibration_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        img_undist = cv2.undistort(gray, K, dist, None, K_undist)
        cv2.imshow('undistorted image', img_undist)
        cv2.waitKey(0)

# ====================================================================================
# load stereo images (Left and Right)
# ====================================================================================
#  set your left and right images
# Example

imgL = cv2.imread()
imgR = cv2.imread()

# convert to grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Convert to undistorted images
# imgLU: undistorted image of imgL
# imgRU: undistorted image of imgR
# grayLU: undistorted image of grayL
# grayRU: undistorted image of grayR
imgLU = np.array([])
imgRU = np.array([])
grayLU = np.array([])
grayRU = np.array([])

# undistorted images
imgLU = cv2.undistort(imgL, K, dist, None, K_undist)
imgRU = cv2.undistort(imgR, K, dist, None, K_undist)
grayLU = cv2.undistort(grayL, K, dist, None, K_undist)
grayRU = cv2.undistort(grayR, K, dist, None, K_undist)

if supervise_mode:
    cv2.imshow('rgb undistorted', cv2.hconcat([imgLU, imgRU]))
    cv2.imshow('gray undistorted', cv2.hconcat([grayLU, grayRU]))

# ====================================================================================
# stereo matching (Dense matching)
# ====================================================================================
# reference: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_depthmap.html

disp8 = np.array([], np.uint8)

stereo = cv2.StereoBM_create(
numDisparities= 16,
blockSize = 33
)
disp8 = stereo.compute(grayLU, grayRU)
disp8 = np.uint8(disp8)

h, w = disp8.shape
disp_max = np.max(disp8)
disp_min = np.min(disp8)

for a in np.arange(h):
    for b in np.arange(w):
        if (disp8[a][b] <= disp_min  or disp8[a][b] >= disp_max ) :
            disp8[a][b] = 0

if supervise_mode:
    imgLU[disp8 < 1, :] = 0
    cv2.imshow('disparity', disp8)
    cv2.imshow('Left Post-processing', imgLU)
    cv2.waitKey(0)

# ====================================================================================
# Visualization
# ====================================================================================
# In advance, you should install open3D (open3d.org)
# pip install open3d

pcd = o3d.geometry.PointCloud()
#  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
#  pc_color: array(Nx3), each row composed with R G,B in the rage of 0~1
pc_points = np.array([], np.float32)
pc_color = np.array([], np.float32)

# 3D reconstruction
# Concatenate pc_points and pc_color

# depth 구하기
h, w = disp8.shape
depth = disp8
for i in np.arange(h):
    for j in np.arange(w):
        depth[i][j] = 255 - disp8[i][j]

# intrinsic param 으로 point cloud 좌표 구하기
fx = K[0][0]
fy = K[1][1]
cx = K[0][2]
cy = K[1][2]

pc_points = np.zeros((h * w, 3))
i = 0
for v in range(h):
    for u in range(w):
        x = (u - cx) * depth[v, u] / fx
        y = (v - cy) * depth[v, u] / fy
        z = depth[v, u]
        pc_points[i] = (x, y, z)
        i = i + 1

# 메서드를 이용한 color 값 찾기
rgb = cv2.cvtColor(imgLU, cv2.COLOR_BGR2RGB)
rgb = rgb.astype(np.float32)/255.0
rgb = rgb.reshape(h*w, 3)
pc_color = rgb

mask = pc_points[:,2] < 255
pc_points = pc_points[mask]
pc_color = pc_color[mask]

#  add position and color to point cloud
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.colors = o3d.utility.Vector3dVector(pc_color)

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.0412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
cv2.destroyAllWindows()

#  end of code