import cv2
import numpy as np


def gaussian(im):
    '''
    Performs the gaussian smoothing on the Input Image
    :param im: Input Image
    :return: Gaussian Smoothed Image
    '''
    # Define the gaussian mask
    mask = np.array([[1, 1, 2, 2, 2, 1, 1],
                     [1, 2, 2, 4, 2, 2, 1],
                     [2, 2, 4, 8, 4, 2, 2],
                     [2, 4, 8, 16, 8, 4, 2],
                     [2, 2, 4, 8, 4, 2, 2],
                     [1, 2, 2, 4, 2, 2, 1],
                     [1, 1, 2, 2, 2, 1, 1]])

    # Calculate the height and width of image
    h, w = im.shape

    # Pad the image with zeros for undefined region
    im_padded = np.zeros((h + 6, w + 6))
    im_padded[3:-3, 3:-3] = im

    # Matrix for holding the smoothed image
    smooth_image = np.zeros(im.shape)

    # Perform the convolution operation

    # If we want to use padded image
    for i in range(h):
        for j in range(w):
            # Multiply the mask matrix with 7*7 matrix and sum the output
            # Divide by output by 140 for normalization
            smooth_image[i, j] = np.sum(mask * im_padded[i:i + 7, j:j + 7]) // 140

    # If we don't want to use padded image
    '''for i in range(3,h-3):
        for j in range(3,w-3):
            # Multiply the mask matrix with 7*7 matrix and sum the output 
            # Divide by output by 140 for normalization 
            smooth_image[i, j] = np.sum(mask * im[i-3:i+4, j-3:j+4]) // 140'''

    return smooth_image


def gradient(im):
    '''
    Compute the gradient magnitude and gradient angle
    of Input Image
    using the Sobel operator.
    :param im: Input Image
    :return: (Gradient Magnitude, Gradient Angle)
    '''
    # Sobel operator in x direction
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    # Sobel operator in y direction
    gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    # Calculate the height and width of image
    h, w = im.shape

    # Define the matrix for storing the gradient magnitude values
    gra_mag = np.zeros(im.shape, dtype=np.float64)

    # Define the matrix for storing the gradient angle value
    gra_angle = np.zeros(im.shape, dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            # Calculate the gradient in x direction
            temp_gx = np.sum(gx * im[i - 1:i + 2, j - 1:j + 2])

            # Calculate the gradient in y direction
            temp_gy = np.sum(gy * im[i - 1:i + 2, j - 1:j + 2])

            # Calculate the gradient magnitude
            gra_mag[i, j] = np.sqrt(np.sum([temp_gx ** 2, temp_gy ** 2]))

            # Calculate the gradient angle
            if temp_gx == 0:
                gra_angle = 0.0
            else:
                gra_angle = np.degrees(np.arctan(temp_gy / temp_gx))

    # If we want to see the output of Gradient Operation
    # cv2.imwrite("G_mag.png",G_mag)

    return (gra_mag, gra_angle)


def NMS(magnitude,angle):
    return -1

if __name__ == '__main__':
    fname = 'Zebra-crossing-1.bmp'
    imname = fname.split('.')[0]

    # Read the image in GRAYSCALE mode
    im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    # Perform the smoothing operation
    smooth_image = gaussian(im)
    gradient, angle = gradient(smooth_image)

    # If we want to test the output of Gaussian Smoothed Image
    # cv2.imwrite('Test_Smooth_{}.png'.format(imname),smooth_image)