import cv2
import numpy as np

def gray_scale(image):
    grayscale_image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return grayscale_image

def gaussian_blur(image, kernel_size=5):
    kernel = gaussian_kernel(kernel_size)
    result = convolution(image, kernel)

    return result.astype(np.uint8)

def gaussian_kernel(kernel_size, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)//2)**2 + (y-(kernel_size-1)//2)**2)/(2*sigma**2)),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)

def sobel_operator_x(image):
    return convolution(image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))

def sobel_operator_y(image):
    return convolution(image, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

def convolution(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    image_padded = np.pad(image, pad, mode='constant')
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(image_padded[i:i+kernel_size, j:j+kernel_size] * kernel)

    return result

def gradient_calc(image):
    sobel_x = sobel_operator_x(image)
    sobel_y = sobel_operator_y(image)
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    gradient_direction = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))

    return gradient_magnitude, gradient_direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    rows, cols = gradient_magnitude.shape
    result = np.zeros_like(gradient_magnitude, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = gradient_direction[i, j] * 180 / np.pi  # Convert radians to degrees
            q, r = 255, 255
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif (22.5 <= direction < 67.5):
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif (67.5 <= direction < 112.5):
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif (112.5 <= direction < 157.5):
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                result[i, j] = gradient_magnitude[i, j]
            else:
                result[i, j] = 0

    return result

def double_threshold(image, low_threshold, high_threshold):
    strong_edges = (image > high_threshold)
    weak_edges = (image >= low_threshold) & (image <= high_threshold)

    return strong_edges, weak_edges

def edge_tracking(strong_edges, weak_edges):
    edges = np.zeros_like(strong_edges)

    strong_edge_coordinates = np.argwhere(strong_edges)
    for i, j in strong_edge_coordinates:
        edges[i, j] = 1

        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if weak_edges[x, y]:
                    edges[x, y] = 1

    return edges

def canny_edge_detection(image, low_threshold, high_threshold):
    image = cv2.imread(image_path)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    gray_image = gray_scale(image)
    blurred_image = gaussian_blur(gray_image)
    gradient_mag, gradient_dir = gradient_calc(blurred_image)
    non_max_suppressed = non_max_suppression(gradient_mag, gradient_dir)
    strong_edges, weak_edges = double_threshold(non_max_suppressed, low_threshold, high_threshold)
    edges = edge_tracking(strong_edges, weak_edges)

    return edges

image_path = 'images/xray_img.jpg'
image = cv2.imread(image_path)
lib_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lib_canny = cv2.Canny(lib_grayscale, 50, 125)
edges = canny_edge_detection(image_path, low_threshold=5, high_threshold=13)
cv2.imshow('Original Image', image)
cv2.imshow('My Canny Edge Detection', (edges * 255).astype(np.uint8))
cv2.imshow('CV2 Canny Edge Detection', lib_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
