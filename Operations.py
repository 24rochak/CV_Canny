import os

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

    # Matrix for holding the smoothed image
    smooth_image = np.zeros(im.shape)

    # Perform the convolution operation on the Non-Padded Image
    for i in range(3, h - 3):
        for j in range(3, w - 3):
            # Multiply the mask matrix with 7*7 matrix and sum the output
            # Divide by output by 140 for normalization
            smooth_image[i, j] = np.sum(mask * im[i - 3:i + 4, j - 3:j + 4]) // 140

    return smooth_image


def gradient_calc(im):
    '''
    Compute the gradient magnitude and gradient angle
    of Input Image using the Sobel operator
    :param im: Input Image
    :return: (Gradient Magnitude, Gradient Angle)
    '''
    # Sobel operator for calculating gradient in x direction
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    # Sobel operator for calculating gradient in y direction
    gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    # Calculate the height and width of image
    h, w = im.shape

    # Define the matrix for storing the Horizontal and Vertical Gradient Values
    G_x = np.zeros(im.shape, dtype=np.float64)
    Normalized_G_x = np.zeros(im.shape, dtype=np.float64)
    G_y = np.zeros(im.shape, dtype=np.float64)
    Normalized_G_y = np.zeros(im.shape, dtype=np.float64)

    # Define the matrix for storing the gradient magnitude values
    Gradient_magnitude = np.zeros(im.shape, dtype=np.float64)
    Normalized_Gradient_magnitude = np.zeros(im.shape, dtype=np.float64)

    # Define the matrix for storing the gradient angle value
    Gradient_angle = np.zeros(im.shape, dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            # Calculate the gradient in x direction
            G_x[i, j] = np.sum(gx * im[i - 1:i + 2, j - 1:j + 2])
            Normalized_G_x[i, j] = G_x[i, j] / 4

            # Calculate the gradient in y direction
            G_y[i, j] = np.sum(gy * im[i - 1:i + 2, j - 1:j + 2])
            Normalized_G_y[i, j] = G_y[i, j] / 4

            # Calculate the gradient magnitude
            Gradient_magnitude[i, j] = np.sqrt((G_x[i, j] ** 2) + (G_y[i, j] ** 2))
            Normalized_Gradient_magnitude[i, j] = np.sqrt(
                (Normalized_G_x[i, j] ** 2) + (Normalized_G_y[i, j] ** 2)) / np.sqrt(2)

            # Calculate the gradient angle
            if G_x[i, j] == 0:
                Gradient_angle[i, j] = 90
            else:
                Gradient_angle[i, j] = np.degrees(np.arctan(G_y[i, j] / G_x[i, j])) + 180

    # Save the Gradients in X direction
    cv2.imwrite("Normalized_G_x.bmp", Normalized_G_x)
    print("Saved Normalized G_x")

    cv2.imwrite("G_x.bmp", G_x)
    print("Saved G_x")

    # Save the Gradients in Y direction
    cv2.imwrite("Normalized_G_y.bmp", Normalized_G_y)
    print("Saved Normalized G_y")

    cv2.imwrite("G_y.bmp", G_y)
    print("Saved G_y")

    # Save the Normalized Gradient Magnitude
    cv2.imwrite("Normalized_Gra_mag.bmp", Normalized_Gradient_magnitude)
    print("Saved Normalized Gradient Magnitude")

    cv2.imwrite("Gra_mag.bmp", Gradient_magnitude)
    print("Saved Gradient Magnitude")

    return (Gradient_magnitude, Gradient_angle, Normalized_Gradient_magnitude)


def NMS(magnitude, angle, normalized_magnitude):
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
    Normalized_N = np.zeros((h, w), dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            # Test if angle is in sector 0
            if (337.5 <= angle[i, j] <= 360) or (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 202.5):
                sector[i, j] = 0
                # Check if the magnitude value is maximum amongst corresponding neighbours
                if magnitude[i, j] > max(magnitude[i, j - 1], magnitude[i, j + 1]):
                    N[i, j] = magnitude[i, j]
                    Normalized_N[i, j] = normalized_magnitude[i, j]
                else:
                    N[i, j] = 0.0
                    Normalized_N[i, j] = 0.0

            # Test if angle is in sector 1
            elif (22.5 <= angle[i, j] < 67.5) or (202.5 <= angle[i, j] < 247.5):
                sector[i, j] = 1
                # Check if the magnitude value is maximum amongst corresponding neighbours
                if magnitude[i, j] > max(magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]):
                    N[i, j] = magnitude[i, j]
                    Normalized_N[i, j] = normalized_magnitude[i, j]
                else:
                    N[i, j] = 0.0
                    Normalized_N[i, j] = 0.0

            # Test if angle is in sector 2
            elif (67.5 <= angle[i, j] < 112.5) or (247.5 <= angle[i, j] < 292.5):
                sector[i, j] = 2
                # Check if the magnitude value is maximum amongst corresponding neighbours
                if magnitude[i, j] > max(magnitude[i - 1, j], magnitude[i + 1, j]):
                    N[i, j] = magnitude[i, j]
                    Normalized_N[i, j] = normalized_magnitude[i, j]
                else:
                    N[i, j] = 0.0
                    Normalized_N[i, j] = 0.0

            # Test if angle is in sector 3
            elif (112.5 <= angle[i, j] < 157.7) or (292.5 <= angle[i, j] < 337.5):
                sector[i, j] = 3
                # Check if the magnitude value is maximum amongst corresponding neighbours
                if magnitude[i, j] > max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]):
                    N[i, j] = magnitude[i, j]
                    Normalized_N[i, j] = normalized_magnitude[i, j]
                else:
                    N[i, j] = 0.0
                    Normalized_N[i, j] = 0.0

    # Save the NMS gradients.
    cv2.imwrite("Normalized_NMS.png", Normalized_N)
    print("Saved Normalized Gradient Magnitude after NMS")

    cv2.imwrite("NMS.png", N)
    print("Saved Gradient Magnitude after NMS")

    return N, sector, Normalized_N


def doubleThresholding(N, angle, t1, t2):
    # Calculate the height and width of NMS matrix
    h, w = N.shape

    # Initialize the Binary Edge Map
    E = np.zeros((h, w))

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            # If Gradient value is less than Threshold 1, assign it to background.
            if N[i, j] < t1:
                E[i, j] = 0

            # If Gradient value is greater than Threshold 2, assign it to Edge.
            elif N[i, j] > t2:
                E[i, j] = 255

            # Gradient value is between Threshold 1 and Threshold 2,
            # check gradient values and angle of 8 neighbours.
            elif t1 <= N[i, j] <= t2:
                if ((N[i - 1, j - 1] > t2 and abs(angle[i - 1, j - 1] - angle[i, j] <= 45)) or  # upper left
                        (N[i - 1, j] > t2 and abs(angle[i - 1, j] - angle[i, j] <= 45)) or  # upper middle
                        (N[i - 1, j + 1] > t2 and abs(angle[i - 1, j + 1] - angle[i, j] <= 45)) or  # upper right
                        (N[i, j - 1] > t2 and abs(angle[i, j - 1] - angle[i, j] <= 45)) or  # left
                        (N[i, j + 1] > t2 and abs(angle[i, j + 1] - angle[i, j] <= 45)) or  # right
                        (N[i + 1, j - 1] > t2 and abs(angle[i + 1, j - 1] - angle[i, j] <= 45)) or  # lower left
                        (N[i + 1, j] > t2 and abs(angle[i + 1, j] - angle[i, j] <= 45)) or  # lower middle
                        (N[i + 1, j + 1] > t2 and abs(angle[i + 1, j + 1] - angle[i, j] <= 45))):  # lower right
                    E[i, j] = 255
                else:
                    E[i, j] = 0
    return E


def trackChanged(x):
    pass


def select_threshold(gradient, angle):
    '''
    Utility to dynamically select the Threshold values
    that are best for the given input image.
    :param gradient: NMS Gradient magnitudes.
    :param angle: Gradient Angles.
    :return: Thresholds t1 and t2
    '''
    # Create a Window for holding Trackbar
    cv2.namedWindow("Threshold", cv2.WINDOW_AUTOSIZE)

    # Create a Trackbar having min value = 0 and max value = 127
    cv2.createTrackbar("T1", "Threshold", 0, 127, trackChanged)

    while True:

        # Get t1 corresponding to the current location of trackbar
        t1 = cv2.getTrackbarPos("T1", "Threshold")
        t2 = 2 * t1

        # Initialize background for displaying current t1 and t2 and text features
        number = np.zeros((200, 500))
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (255, 255, 255)

        # Display values of t1 and t2 on background
        cv2.putText(number, text="T1: " + str(t1), org=(100, 100), fontFace=font, fontScale=1, color=color, thickness=1)
        cv2.putText(number, text="T2: " + str(int(t1) * 2), org=(300, 100), fontFace=font, fontScale=1, color=color,
                    thickness=1)

        # Calculate the edge map using current t1 and t2
        edge_map = doubleThresholding(gradient, angle, t1, t2)

        # Display the edge map and the threshold window
        cv2.imshow("Edge_Map", edge_map)
        cv2.imshow("Threshold", number)

        # When satisfied with result, press key='q' to finalize t1 and t2
        if cv2.waitKey(1) == ord('q'):
            break

    # Return selected t1 and t2
    return (t1, t2)


if __name__ == '__main__':

    # Specify the filename of the Input Image
    fname = 'Houses-225.bmp'

    # Read the image in GRAYSCALE mode
    im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    # Create a Directory to store the results.
    if not os.path.exists("Output_" + fname):
        os.mkdir("Output_" + fname)
    os.chdir("Output_" + fname)

    print("Performing Gaussian Smoothing on {}".format(fname))
    smooth_image = gaussian(im)
    print("Gaussian Smoothing completed")

    cv2.imwrite('Smooth_{}'.format(fname), smooth_image)
    print("Saved Smoothed Image\n")

    print("Performing Gradient Calculation on {}".format(fname))
    gradient, angle, normalized_gradient = gradient_calc(smooth_image)
    print("Gradient Calculation completed\n")

    print("Performing Non-Maximum Suppression on {}".format(fname))
    NMS_gradient, sector, Normalized_NMS_gradient = NMS(gradient, angle, normalized_gradient)
    print("Non-Maximum Suppression completed\n")

    print("Please select the Thresholds")
    print("Move the slider around to select the threshold values that are best for the current Image")
    print("Press key 'q' when the edge map is acceptable")
    print("DO NOT PRESS CLOSE BUTTON {X} TO CLOSE THE WINDOW")
    t1, t2 = select_threshold(Normalized_NMS_gradient, angle)
    print("Selected Thresholds-> T1:{}\tT2:{}\n".format(t1, t2))

    print("Performing Double Thresholding on {}".format(fname))
    Edge_map = doubleThresholding(Normalized_NMS_gradient, angle, t1=t1, t2=t2)
    print("Double Thresholding completed")

    cv2.imwrite('EdgeMap_T1_{}_T2_{}_{}'.format(t1, t2, fname), Edge_map)
    print("Saved EdgeMap")
