import numpy as np
import cv2
from lxml import etree
from lxml.builder import E


def px_to_mm(px):

    #Read it from file...
    #rotationMatrix_inv = 
    #cameraMatrix_inv = 
    #scalar = 
    #tvec = 

    mm = np.dot(rotationMatrix_inv,(np.dot( cameraMatrix_inv, scalar*px ) - tvec))
    return mm.ravel()

pts_row = 9
pts_col = 6

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # online
objp = np.zeros((pts_row*pts_col,3), np.float32) # [ [0,0,0] [0,0,0] ... ]
objp[:,:2] = np.mgrid[0:pts_row,0:pts_col].T.reshape(-1,2)
objpoints = []
imgpoints = []
d=1

while True:
    
    if d == 26:
        break

    img = cv2.imread("picture/file_%d.jpg"%d)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (pts_row,pts_col)) #none is optional

    if ret == True:
        d+=1
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (pts_row,pts_col), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    else:
        d+=1
        print("Try Again")

retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, None)
print("Error: ")
print(retval)
# #The reference
# img = cv2.imread("images/file_0.jpg")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, pt = cv2.findChessboardCorners(gray, (pts_row,pts_col))
# pts = cv2.cornerSubPix(gray,pt,(11,11),(-1,-1),criteria)
# print(pts)

imagePoints = np.array([    (986.0,601.0),
                            (1048.0,601.0),
                            (986.0,540.0),
                            (986.0,478.0),
                            (1109.0,601.0)       ])

objectPoints = np.array([   (0.0,0.0,0.0),
                            (50.0,0.0,0.0),
                            (0.0,50.0,0.0),
                            (0.0,100.0,0.0),
                            (100.0,0.0,0.0)     ])
retval, rvec, tvec = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
rotationMatrix,_ = cv2.Rodrigues(rvec)
cameraMatrix_inv = np.linalg.inv(cameraMatrix) #check
rotationMatrix_inv = np.linalg.inv(rotationMatrix)
temp = np.array([[0],[0],[1]])
scalar = np.dot(cameraMatrix, np.dot(rotationMatrix,temp) + tvec )
scalar = scalar[2][0] #get 3rd value

# dist_coeff_xml = E.distCoeffs(*map(E.coef, map(str, distCoeffs.ravel())))
# camera_matrix_xml = E.cameraMatrix(*map(E.data, map(str, cameraMatrix.ravel())))
# xmldoc = E.calibration(camera_matrix_xml, dist_coeff_xml)
# fname = "data.xml"
# f = open(fname, "wb")
# f.write(etree.tostring(xmldoc, pretty_print=True))

#store values:
#calibrationmatrix
#distributioncoefficient

#translationalvector
#scalar
#cameramatrix_inv
#rotationmatrix_inv

#px_to_mm(point)

point = np.array([[1233],[447],[1]]) #check
print(point)
mm4 = np.dot(rotationMatrix_inv,(np.dot( cameraMatrix_inv, scalar*point ) - tvec))
print(mm4)

#main
    #input("calibrate camera? [y/n]: ")
        #calibrate()
        #set_origin()
    #input("set origin? [y/n]: ")
        #set_origin()
    #input("convert? [y/n]: ")
        #input("Enter x y coordinate: ")
        #mm = px_to_mm()
        #print(mm)