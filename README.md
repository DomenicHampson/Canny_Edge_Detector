# Canny Edge Detection in Python #
This project implements the Canny Edge Detection algorithm in Python using OpenCV and NumPy. Canny Edge Detection is a popular algorithm used in image processing to detect a wide range of edges in images.

## Features ##
- Grayscale conversion: Converts the input image to a grayscale image.
- Gaussian blurring: Applies Gaussian blur to the grayscale image to reduce noise.
- Gradient calculation: Calculates the gradient magnitude and direction of the blurred image using Sobel operators.
- Non-maximum suppression: Suppresses non-maximum pixels to thin out the edges.
- Double thresholding: Applies double thresholding to classify edge pixels as strong, weak, or non-edge pixels.
- Edge tracking by hysteresis: Tracks edges by connecting strong edges and weak edges that are connected to strong edges.
