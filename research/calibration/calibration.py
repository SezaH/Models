import numpy as np
import cv2
from lxml import etree
from lxml.builder import E
import xml.etree.ElementTree as ET

def calibration():
    pts_row = 9
    pts_col = 6

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((pts_row*pts_col,3), np.float32)
    objp[:,:2] = np.mgrid[0:pts_row,0:pts_col].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    d=1
    cap = cv2.VideoCapture(0)

    #Take pics
    while True:
        
        if d == 26:
            break

        input("press enter")

        ret,img = cap.read()
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
            #d+=1
            print("Try Again")

    #Calibrate
    print("calibrating...")
    retval, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, None)
    
    #Save file
    print("Saving...")
    dist_coeff_xml = E.distCoeffs(*map(E.data, map(str, distCoeffs.ravel())))
    camera_matrix_xml = E.cameraMatrix(*map(E.data, map(str, cameraMatrix.ravel())))
    xmldoc = E.calibration(camera_matrix_xml, dist_coeff_xml)
    fname = "data.xml"
    f = open(fname, "wb")
    f.write(etree.tostring(xmldoc, pretty_print=True))
    f.close()

def set_coordinates():
    pts_row = 9
    pts_col = 6
    distance = 50

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #Read data from file
    #if file not found, return false and ask to do calibration.
    tree = ET.parse("data.xml")
    root = tree.getroot()

    cm = root.find("cameraMatrix")
    counter = 1
    cameraMatrix = np.empty((0,3))
    row = np.empty((0,3))
    for data in cm.findall("data"):
        if (counter % 3) == 0:
            row = np.append(row,[float(data.text) ])
            row = np.reshape(row, (-1, 3))
            cameraMatrix = np.append( cameraMatrix, row, axis = 0 )
            row = np.empty((0,3))
        else:
            row = np.append(row,[ float(data.text) ])
        counter+=1

    dc = root.find("distCoeffs")
    distCoeffs = np.empty((0,1))
    for data in dc.findall("data"):
            distCoeffs = np.append(distCoeffs,[ float(data.text) ])
    distCoeffs = np.reshape(distCoeffs, (-1, distCoeffs.size))

    #Take one pic, with corrdinates
    #The reference
    img = cv2.imread("picture/file_0.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, pt = cv2.findChessboardCorners(gray, (pts_row,pts_col))
    pts = cv2.cornerSubPix(gray,pt,(11,11),(-1,-1),criteria)

    half_row = int((pts_row+1)/2)
    half_col = int((pts_col+1)/2)

    img_center = pts[ (half_col - 1)*pts_row + half_row - 1][0]
    img_top = pts[half_row - 1][0]
    img_bottom = pts[ (pts_col - 1)*pts_row + half_row - 1][0]
    img_left = pts[ (half_col - 1)*pts_row][0]
    img_right = pts[ half_col*pts_row - 1][0]

    #obj
    obj_top = -1 * distance * int((pts_col - 1)/2)
    obj_bottom = distance * int((pts_col)/2)
    obj_left = -1 * distance * int((pts_row - 1)/2)
    obj_rigth = distance * int((pts_row)/2)


    imagePoints = np.array([    (img_center[0],img_center[1]),	#center
                                (img_top[0],img_top[1]),		#top
                                (img_bottom[0],img_bottom[1]),	#bottom
                                (img_left[0],img_left[1]),		#left
                                (img_right[0],img_right[1]) ])	#right

    objectPoints = np.array([   (0.0,0.0,0.0),			#center
                                (0.0,obj_top,0.0),		#top
                                (0.0,obj_bottom,0.0),	#bottom
                                (obj_left,0.0,0.0),		#left
                                (obj_rigth,0.0,0.0)	])	#right

    #get values for formula
    retval, rvec, tvec = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
    rotationMatrix,_ = cv2.Rodrigues(rvec)
    rotationMatrix_inv = np.linalg.inv(rotationMatrix)
    cameraMatrix_inv = np.linalg.inv(cameraMatrix) #check
    temp = np.array([[0],[0],[1]]) #z = 1 or 0, idk
    scalar = np.dot(cameraMatrix, np.dot(rotationMatrix,temp) + tvec )
    scalar = scalar[2][0] #get 3rd value

    #save values on file
    rotationMatrixInv_xml = E.rotationMatrixInv(*map(E.data, map(str, rotationMatrix_inv.ravel())))
    cameraMatrixInv_xml = E.cameraMatrixInv(*map(E.data, map(str, cameraMatrix_inv.ravel())))
    tVec_xml = E.tVec(*map(E.data, map(str, tvec.ravel())))
    scalar_xml = E.scalar(*map(E.data, map(str, [scalar])))
    xmldoc = E.origin(rotationMatrixInv_xml, cameraMatrixInv_xml)
    xmldoc.append(tVec_xml)
    xmldoc.append(scalar_xml)
    fname = "origin.xml"
    f = open(fname, "wb")
    f.write(etree.tostring(xmldoc, pretty_print=True))
    f.close()
    return True

def px_to_mm():
    px = np.array([[986],[600],[1]]) #check

    #Read it from file...
    tree = ET.parse("origin.xml")
    root = tree.getroot()

    rm = root.find("rotationMatrixInv")
    counter = 1
    rotationMatrixInv = np.empty((0,3))
    row = np.empty((0,3))
    for data in rm.findall("data"):
        if (counter % 3) == 0:
            row = np.append(row,[float(data.text) ])
            row = np.reshape(row, (-1, 3))
            rotationMatrixInv = np.append( rotationMatrixInv, row, axis = 0 )
            row = np.empty((0,3))
        else:
            row = np.append(row,[ float(data.text) ])
        counter+=1

    # cameraMatrix_inv = 
    cmi = root.find("cameraMatrixInv")
    counter = 1
    cameraMatrixInv = np.empty((0,3))
    row = np.empty((0,3))
    for data in cmi.findall("data"):
        if (counter % 3) == 0:
            row = np.append(row,[float(data.text) ])
            row = np.reshape(row, (-1, 3))
            cameraMatrixInv = np.append( cameraMatrixInv, row, axis = 0 )
            row = np.empty((0,3))
        else:
            row = np.append(row,[ float(data.text) ])
        counter+=1

    # scalar
    sc = root.find("scalar")
    scalar = float(sc.find("data").text )

    # tvec
    tv = root.find("tVec")
    tVec = np.empty((0,1))
    for data in tv.findall("data"):
            tVec = np.append(tVec,[ float(data.text) ] )
    tVec = np.reshape(tVec, (-1, tVec.size))
    tVec = np.swapaxes(tVec,0,1)

    mm = np.dot(rotationMatrixInv,(np.dot( cameraMatrixInv, scalar*px ) - tVec))

    return mm

def main():
	#row,col,
    #calibration()
    set_coordinates()
    px_to_mm()


if __name__ == "__main__":
    main()