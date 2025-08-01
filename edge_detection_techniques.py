import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image_gray(image_path='image.jpg'):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def show_image(title, image):
    plt.figure(figsize=(6, 4))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def canny_edge(img):
    edges = cv2.Canny(img, 100, 200)
    show_image("Canny Edge", edges)

def sobel_edge(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    show_image("Sobel Edge", sobel)

def laplacian_edge(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))
    show_image("Laplacian Edge", laplacian)

def prewitt_edge(img):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)
    prewitt = cv2.magnitude(np.float32(img_prewittx), np.float32(img_prewitty))
    prewitt = np.uint8(np.clip(prewitt, 0, 255))
    show_image("Prewitt Edge", prewitt)

if __name__ == "__main__":
    img = load_image_gray()

    canny_edge(img)
    sobel_edge(img)
    laplacian_edge(img)
    prewitt_edge(img)
