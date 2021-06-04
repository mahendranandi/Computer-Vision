import numpy as np
#### DO NOT IMPORT cv2 

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  filtered_image=image.copy()
  im_height=image.shape[0]  # height of the image
  im_width=image.shape[1]   # width of the image
  padh=int((filter.shape[0])/2)   # size of the pad in the direction of the height of the image
  padw=int((filter.shape[1])/2)   # size of the pad in the direction of the width of the image
  padded_matrix=np.pad(filtered_image,((padh,padh),(padw,padw),(0,0)),'reflect')# padding the matrix with reflect type
  for d in range(image.shape[2]): # for loop for each channel filtering
    for i in range(image.shape[0]): #for loop for accessing the row index i.e., i-th index of any arbitrary (i,j)-th point point of the image
        for j in range(image.shape[1]): #for loop for accessing the column index i.e., j-th index of any arbitrary (i,j)-th point of the image
            filtered_image[i][j][d] = np.sum(np.multiply(padded_matrix[i:i+filter.shape[0],j:j+filter.shape[1],d],filter)) #filtered output on (i,j)-th pixel of each channel
  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
  low_frequencies=my_imfilter(image1,filter) #creating a low-frequency image
  high_frequencies =image2 - my_imfilter(image2, filter) #removing the low frequency componenets from the other image
  hybrid_image = 0.5*low_frequencies + 0.5*high_frequencies #creating hybrid imges
  hybrid_image=np.clip(hybrid_image,0.0,1.0) # handling the overflow of pixel values
  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
