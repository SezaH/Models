import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Get camera
vid = cv2.VideoCapture(0)

# Instructions
print("[P] to take a pic")
print("[Q] to exit")

# To store the images you took, can be removed if wanted, not important
d=1

# In each loop, gets a pic and store the corner points.
while True:
    key = input("Enter: ")

    if key == "Q" or key == "q":
        break
    
    # Get image from camera
    #ret, img = vid.read() #To use the camera
    img = cv2.imread("images/file_%d.jpg"%d) # To upload images

    # Convert it to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        
        #save image in directory
        print("Good Pic! #%d"%d)
        #filename = "images/file_%d.jpg"%d
        #cv2.imwrite(filename, img)
        d+=1
        
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    else:
        print("Try Again")

#Calibrate and store the matrix
print("Calibrating all pictures")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
if ret == True:
    print("Calibrated!")
else:
    print("error")

print(dist)

print(ret)
print(mtx)
cv2.destroyAllWindows()