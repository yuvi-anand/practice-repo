import sys
from this import d
import numpy as np
import cv2
import os


#######CALIBRATION OF INDIVUDUAL CAMERAS####### 




# Set the path to the images captured by the left and right cameras
pathR = r"C:\Users\yuvia\Downloads\checkerboard\checkerboard\left_camera"
pathL = r"C:\Users\yuvia\Downloads\checkerboard\checkerboard\right_camera"
 
# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#filesL = a list of file names within directory 


###filesL  = [os.listdr("C:\Users\yuvia\Downloads\checkerboard\checkerboard\left_camera")]


# filesL = os.listdir(pathL)
# filesR = os.listdir(pathR)

# #filesR = a list of file names within directory 



# if (len(filesL) != len(filesR)):
#     sys.exit("different number of left and right images")


 
# objp = np.zeros((14*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:14,0:7].T.reshape(-1,2)
 
# img_ptsL = []
# img_ptsR = []
# obj_pts = []
 
# scale = 0.2

# for i in range(len(filesL)): #replace tqdm with len of files  
    
#   print(filesL[i])

    
#   imgL = cv2.imread( os.path.join(pathL, filesL[i])) #combining os path from path l with the file name
#   imgR = cv2.imread(os.path.join(pathR, filesR[i])) #combining os path from path r with the file name 
#   imgL= cv2.resize(imgL,(0,0), fx = scale, fy = scale)
#   imgR =  cv2.resize(imgR,(0,0), fx = scale, fy = scale)
#   imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#   imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
 
#   outputL = imgL.copy()
#   outputR = imgR.copy()

#   retR, cornersR =  cv2.findChessboardCorners(imgR_gray,(14,7),None)
#   retL, cornersL = cv2.findChessboardCorners(imgL_gray,(14,7),None)

#   if retR and retL:
#     obj_pts.append(objp)
#     cornersR = cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
#     cornersL = cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
#     cv2.drawChessboardCorners(outputR,(14,7),cornersR,retR)
#     cv2.drawChessboardCorners(outputL,(14,7),cornersL,retL)
#     detection = np.hstack((outputL, outputR))
#     cv2.imwrite(str(i) + ".png", detection)
#     img_ptsL.append(cornersL)
#     img_ptsR.append(cornersR)

 
# # Calibrating left camera
# retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
# hL,wL= imgL_gray.shape[:2]
# new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))
 
# # Calibrating right camera
# retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
# hR,wR= imgR_gray.shape[:2]
# new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

# print(retL)
# print(retR)

# ######STEREO CALIBRATION#######


# flags = 0
# flags |= cv2.CALIB_FIX_INTRINSIC
# # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# # Hence intrinsic parameters are the same
 
# criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
 
# # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
# retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], criteria_stereo, flags)


# print(retS)

# ######STEREO RECTIFICATION#####

# rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, (3000, 4096)  )

# ######MAPPING COMPUTATION#####

# Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
#                                              imgL_gray.shape[::-1], cv2.CV_16SC2)
# Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
#                                               imgR_gray.shape[::-1], cv2.CV_16SC2)
 
# print("Saving paraeters ......")
# cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
# cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
# cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
# cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
# cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
# cv_file.release()


imgL = cv2.imread("tsukuba_L.png")
imgR = cv2.imread("tsukuba_R.png")


# left_rectified = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
# right_rectified = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

# preCalib = np.hstack((imgL, imgR))
# postCalib = np.hstack((left_rectified, right_rectified))
# cv2.imwrite("preCalib.png", preCalib)
# cv2.imwrite("postCalib.png", postCalib)

scale_percent = 100

width = (imgL.shape[1] * scale_percent/100)
height = (imgL.shape[0] * scale_percent/100)
dim = (width,height)



imgL_small = cv2.resize(imgL, np.uint16(dim), interpolation = cv2.INTER_LINEAR)
imgR_small = cv2.resize(imgR, np.uint16(dim), interpolation = cv2.INTER_LINEAR)

imgL_small = cv2.cvtColor(imgL_small, cv2.COLOR_BGR2GRAY)
imgR_small = cv2.cvtColor(imgR_small, cv2.COLOR_BGR2GRAY)



stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(imgL_small,imgR_small)

print(np.max(disparity))
print(np.min(disparity))


disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

print(np.max(disparity))
print(np.min(disparity))

cv2.imwrite('disparity.png', disparity)

