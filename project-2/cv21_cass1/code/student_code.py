import numpy as np
import cv2 # You must not use any methods which has 'hough' in it!
from utils import  hough_peaks




def hough_lines_vote_acc(edge_img, rho_res=1, thetas= np.arange(0,180)):
    """
      Creating an Hough Vote Accumulator. The Generated grid will have theatas on 
      X axis, and rhos on the Y axis. 

      Args
      - edge_img: numpy nd-array of dim (m, n)
      - rho_res: rho resolution. Default is 1. This controls how dense your grid columns are
      - thetas: theta values. 
        
      Returns
      - accumulator with votes for each grid point, thetas, and rhos

      HINTS:
      - I encourage you to try implementing this naively first, just be aware that
       it may take an absurdly long time to run. You will need to get a function
       that takes a reasonable amount of time to run so that I can verify
       your code works.
    """

    
    ############################
    ### TODO: YOUR CODE HERE ###
    max_rho=int(np.sqrt(edge_img.shape[0]**2+edge_img.shape[1]**2))
    rhos=np.arange(-max_rho,max_rho)
    accumulator=np.zeros([int(len(rhos)),len(thetas)])
    thetas1=np.deg2rad(thetas)
    for i in range(edge_img.shape[0]):
        for j in range(edge_img.shape[1]):
            if (edge_img[i,j]==255):
                for t in range(len(thetas)):
                    
                    rho=int(j*np.cos(thetas1[t]) + i*np.sin(thetas1[t]))
                    accumulator[rho+max_rho,np.int(np.round(np.rad2deg(thetas1[t])))] +=1
                 

    ### END OF STUDENT CODE ####
    ############################
    return accumulator, thetas, rhos

    
def hough_circles_vote_acc(edge_img, radius):
    """
      Creating an Hough Vote Accumulator. The Generated grid will have 
      x coordinate of the center of cirlce on 
      X axis, and y coordinate of the center of cirlces on the Y axis. 

      Args
      - edge_img: numpy nd-array of dim (m, n)
      - radius: radius of the circle
        
      Returns
      - accumulator with votes for each grid point

      HINTS:
      - I encourage you to try implementing this naively first, just be aware that
       it may take an absurdly long time to run. You will need to get a function
       that takes a reasonable amount of time to run so that I can verify
       your code works.
    """
  
    
    ############################
    ### TODO: YOUR CODE HERE ###
    thetas=np.arange(-180,180,1)
    r=radius
    accumulator=np.zeros([edge_img.shape[0],edge_img.shape[1]])
    for i in range(edge_img.shape[1]):
        for j in range(edge_img.shape[0]):
            if edge_img[j,i]==255:
                for t in range(len(thetas)):
                    a=int(i+r*np.cos(thetas[t]))
                    b=int(j-r*np.sin(thetas[t]))
                    if (a>=0 and a<edge_img.shape[1] and b>=0 and b<edge_img.shape[0]) :
                        accumulator[b,a] +=1
                    

    ### END OF STUDENT CODE ####
    ############################        
    return accumulator



def find_circles(edge_img, radius_range=[1,2], threshold=100, nhood_size=10):
    """
      A naive implementation of the algorithm for finding all the circles in a range.
      Feel free to write your own more efficient method [Extra Credit]. 
      For extra credit, you may need to add additional arguments. 


      Args
      - edge_img: numpy nd-array of dim (m, n). 
      - radius_range: range of radius. All cicles whose radius falls 
      in between should be selected.
      - nhood_size: size of the neighborhood from where only one candidate can be chosen. 
      
      Returns
      - centers, and radii i.e., (x, y) coordinates for each circle.

      HINTS:
      - I encourage you to use this naive version first. Just be aware that
       it may take a long time to run. You will get EXTRA CREDIT if you can write a faster
       implementaiton of this method, keeping the method signature (input and output parameters)
       unchanged. 
    """
    n = radius_range[1] - radius_range[0]
    H_size = (n,) + edge_img.shape
    H = np.zeros(H_size, dtype=np.uint)
    centers = ()
    radii = np.arange(radius_range[0], radius_range[1])
    valid_radii = np.array([], dtype=np.uint)
    num_circles = 0
    for i in range(len(radii)):
        H[i] = hough_circles_vote_acc(edge_img, radii[i])
        peaks = hough_peaks(H[i], numpeaks=10, threshold=threshold,
                            nhood_size=nhood_size)
        if peaks.shape[0]:
            valid_radii = np.append(valid_radii, radii[i])
            centers = centers + (peaks,)
            for peak in peaks:
                cv2.circle(edge_img, tuple(peak[::-1]), radii[i]+1, (0,0,0), -1)
        #  cv2.imshow('image', edge_img); cv2.waitKey(0); cv2.destroyAllWindows()
        num_circles += peaks.shape[0]
        print('Progress: %d%% - Circles: %d\033[F\r'%(100*i/len(radii), num_circles))
    print('Circles detected: %d          '%(num_circles))
    centers = np.array(centers)
    return centers, valid_radii.astype(np.uint)
