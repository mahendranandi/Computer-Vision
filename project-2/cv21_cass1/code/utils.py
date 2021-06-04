import cv2
import numpy as np
import matplotlib.pyplot as plt

def clip(idx):
    return int(max(idx,0))


def preprocess(image, erodeK, blurK, blurSigma, lowT, upT):
    """
      Preprocess an image by eroding (opt.), blurring (opt.), and then applying
      Canny edge detector. 

      Args
      - image: numpy nd-array of dim (m, n, c)
      - erodeK: size of kernal for erode operation. Dimension: (erodeK, erodeK)
      - blurK: size of kernal for gaussing blur. Dimension: (blurK, blurK) 
      - blurSigma: sigma used for applying gaussian blur. 
      - lowT: low threshold value for Canny operator.
      - highT: high threshold value for Canny operator.  
      Returns
      - edge-image: numpy nd-array of dim (m, n)

      HINTS:
      - Apply your own preprocessing, if required. Shortly describe how your implementation
        is different, in the writeup. 
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eroded_img = cv2.erode(gray_img, np.ones((erodeK,erodeK),np.uint8), 1)
    smoothed_img = cv2.blur(eroded_img, (blurK,blurK), blurSigma)    
    # Canny
    edged = cv2.Canny(smoothed_img, lowT, upT)
    return edged
    

def hough_peaks(H, numpeaks=1, threshold=100, nhood_size=5):
    """
      Returns the top numpeaks from the accumulator H

      Args
      - H: Hough Space (Voting Accumulator)
      - numpeaks: Number of peaks to return
      - threshold: Minimum number of votes to get considered for picked
      - nhood_size: neighborhood size. Only one peak will be chosen from 
        any neighborhood.   
      Returns
      - peak coordinates: numpy nd-array of dim (numpeaks, 2)


    """
    peaks = np.zeros((numpeaks,2), dtype=np.uint64)
    temp_H = H.copy()
    for i in range(numpeaks):
        _,max_val,_,max_loc = cv2.minMaxLoc(temp_H) # find maximum peak
        if max_val > threshold:
            peaks[i] = max_loc
            (c,r) = max_loc
            t = nhood_size//2.0
            temp_H[clip(r-t):int(r+t+1), clip(c-t):int(c+t+1)] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks[:,::-1]
    

def hough_lines_draw(img, outfile, peaks, rhos, thetas):
    """
      Returns the image with hough lines drawn.
      Args
      - img: Image on which lines will be drawn
      - outfile: The output file. The file will be saved. 
      - peaks: peaks returned by hough_peaks
      - rhos: array of rhos used in Hough Space  
      - thetas: array of thetas used in Hough Space
      Returns
      - img: after drwaing lines on it.

    """
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]] * np.pi / 180.0
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite(outfile, img)
    return img


def hough_circles_draw(img, outfile, peaks, radius):
    """
      Returns the image with hough circles drawn.
      Args
      - img: Image on which circles will be drawn
      - outfile: The output file. The file will be saved. 
      - peaks: peaks returned by hough_peaks. Contails tuple of (y, x) coordinates. 
      - radius: radius of the circle  
      Returns
      - img: after drwaing circles on it.

    """
    for peak in peaks:
        cv2.circle(img, tuple(peak[::-1]), radius, (0,255,0), 2)
    cv2.imwrite(outfile, img)
    return img




def im2single(im):
  im = im.astype(np.float32) / 255
  return im

def single2im(im):
  im *= 255
  im = im.astype(np.uint8)
  return im


def load_image(path):
  return im2single(cv2.imread(path))[:, :, ::-1]

def save_image(path, im):
  return cv2.imwrite(path, single2im(im.copy())[:, :, ::-1])


def load_image_u8(path):
  return cv2.imread(path)[:, :, ::-1]

def save_image_u8(path, im):
  return cv2.imwrite(path, im[:, :, ::-1])




def plotFigure(image, dimension=3):

    # display the images in RGB. 

    plt.figure(figsize=(dimension,dimension))
    plt.imshow(image[:,:,::-1])
    return


def plotFigureGray(image, dimension=3):

    # display the image in grayscale. 

    plt.figure(figsize=(dimension,dimension))
    plt.imshow(image,cmap='gray')
    return






