import cv2
import numpy as np
import matplotlib.pyplot as plt

def mapp(h):                                                                  #aligning the four boundary points in our desired order.
    h = h.reshape((4,2))
    hnew = np.zeros((4,2), dtype = np.float32)
    a=[]
    b=[]
    
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    
    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
#     for i in range(4):
#         if (i != np.argmin(add) and i != np.argmax(add)):
#             hnew[k] = h[i]
#             k = 3
    
    
    return hnew

def camscan(image, epsilon):                                               
    orig = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                              #grayscale conversion
    plt.figure(figsize = (20,10))
    plt.subplot(141)
    plt.imshow(gray,'gray')
    plt.title("grayscale image")
    
    blurred = cv2.GaussianBlur(gray,(5,5),0) 
    plt.subplot(142)
    plt.imshow(blurred,'gray')
    plt.title("blurred image")
    
    edges = cv2.Canny(blurred,30,50)                                                            #finding edges 
    plt.subplot(143)
    plt.imshow(edges,'gray')
    plt.title("edge image")
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)       #finding the boundary of the page
    contours = sorted(contours, key = cv2.contourArea, reverse=True)
    for c in contours:
        p = cv2.arcLength(c,True)                                                               #tries to find the square
        approx = cv2.approxPolyDP(c, epsilon*p, True)                                           #approximates the curve
        if len(approx) == 4:
            target = approx
            break
    approx = mapp(target)
    print("before allignment : ")
    print(target)
    print("after allignment : ")
    print(approx)
    
    pts = np.float32([[0,0],[300,0],[300,400],[0,400]])
    homographyMat, status = cv2.findHomography(approx, pts)
    #op = cv2.getPerspectiveTransform(approx,pts)                                                    #birdeye view of the contour
    dst = cv2.warpPerspective(orig,homographyMat,(300,400))
    
    plt.subplot(144)
    plt.imshow(dst,'gray')
    plt.title("final cropped image")
    
    return dst,gray,blurred,edges 

        
    