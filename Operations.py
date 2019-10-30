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

    # Perform the convolution operation on the padded Image
    for i in range(h):
        for j in range(w):
            # Multiply the mask matrix with 7*7 matrix and sum the output
            # Divide by output by 140 for normalization
            smooth_image[i, j] = np.sum(mask * im_padded[i:i + 7, j:j + 7]) // 140

    # Perform the convolution operation on the Non-Padded Image
    '''for i in range(3,h-3):
        for j in range(3,w-3):
            # Multiply the mask matrix with 7*7 matrix and sum the output 
            # Divide by output by 140 for normalization 
            smooth_image[i, j] = np.sum(mask * im[i-3:i+4, j-3:j+4]) // 140'''

    return smooth_image


def gradient(im):
    '''
    Compute the gradient magnitude and gradient angle
    of Input Image using the Sobel operator
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
                gra_angle[i, j] = 0.0
            else:
                gra_angle[i, j] = np.degrees(np.arctan(temp_gy / temp_gx))

    return (gra_mag, gra_angle)


def NMS(magnitude, angle):
    '''
    Performs Non-Maximum Suppression on given Magnitude matrix
    using angle matrix
    :param magnitude: Input Magnitude matrix
    :param angle: Corresponding Angle matrix
    :return: Non-Maximum Suppressed Magnitude and Corresponding Sector matrices
    '''

    # Calculate height, width of the Magnitude Matrix
    h, w = im.shape

    # Initialize the sector matrix to store the sector values.
    sector = np.zeros((h, w))

    # Initialize the output matrix
    N = np.zeros((h, w), dtype=np.float64)

    print(h, w)

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            # Test if angle is in sector 0
            if (337.5 <= angle[i, j] <= 360) or (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 202.5):
                sector[i, j] = 0
                # Check if the magnitude value is maximum amongst corresponding neighbours
                if magnitude[i, j] > max(magnitude[i, j - 1], magnitude[i, j + 1]):
                    N[i, j] = magnitude[i, j]
                else:
                    N[i, j] = 0.0

            # Test if angle is in sector 1
            elif (22.5 <= angle[i, j] < 67.5) or (202.5 <= angle[i, j] < 247.5):
                sector[i, j] = 1
                # Check if the magnitude value is maximum amongst corresponding neighbours
                if magnitude[i, j] > max(magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]):
                    N[i, j] = magnitude[i, j]
                else:
                    N[i, j] = 0.0

            # Test if angle is in sector 2
            elif (67.5 <= angle[i, j] < 112.5) or (247.5 <= angle[i, j] < 292.5):
                sector[i, j] = 2
                # Check if the magnitude value is maximum amongst corresponding neighbours
                if magnitude[i, j] > max(magnitude[i - 1, j], magnitude[i + 1, j]):
                    N[i, j] = magnitude[i, j]
                else:
                    N[i, j] = 0.0

            # Test if angle is in sector 3
            elif (112.5 <= angle[i, j] < 157.7) or (292.5 <= angle[i, j] < 337.5):
                sector[i, j] = 3
                # Check if the magnitude value is maximum amongst corresponding neighbours
                if magnitude[i, j] > max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]):
                    N[i, j] = magnitude[i, j]
                else:
                    N[i, j] = 0.0

    return N, sector


if __name__ == '__main__':
    fname = 'Zebra-crossing-1.bmp'

    # Read the image in GRAYSCALE mode
    im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    # Perform the smoothing operation

    print("Performing Gaussian Smoothing on {}".format(fname))
    smooth_image = gaussian(im)
    print("Gaussian Smoothing completed")

    print("Performing Gradient Calculation on {}".format(fname))
    gradient, angle = gradient(smooth_image)
    print("Gradient Calculation completed")

    print("Performing Non-Maximum Suppression on {}".format(fname))
    NMS_gradient, sector = NMS(gradient, angle)
    print("Non-Maximum Suppression completed")

    cv2.imwrite("NMS_{}".format(fname), NMS_gradient)
    print("Saved Image")

    # If we want to test the output of Gaussian Smoothed Image
    # cv2.imwrite('Test_Smooth_{}'.format(fname),smooth_image)
