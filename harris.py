import cv2
import numpy as np
from scipy import ndimage
import math


def harris_corner_detector(img):
    height,width = img.shape[:2]

    sobel = np.zeros((height, width, 2), dtype=np.float32)
    sobel[:,:,0] = cv2.Sobel(img,cv2.CV_16S,1,0,3)
    sobel[:,:,1] = cv2.Sobel(img,cv2.CV_16S,0,1,3)

    Ix = np.zeros((height, width, 3), dtype=np.float32)
    Iy = np.zeros((height, width, 3), dtype=np.float32)

    Ix[:,:,0] = sobel[:,:,0]*sobel[:,:,0]
    Ix[:,:,1] = sobel[:,:,1]*sobel[:,:,1]
    Ix[:,:,2] = sobel[:,:,0]*sobel[:,:,1]

    Iy[:,:,0] = cv2.GaussianBlur(Ix[:,:,0],(3,3),2)
    Iy[:,:,1] = cv2.GaussianBlur(Ix[:,:,1],(3, 3),2)
    Iy[:,:,2] = cv2.GaussianBlur(Ix[:,:,2],(3, 3),2)

    IxIy = [np.array([[Iy[i,j,0],Iy[i,j,2]],[Iy[i,j,2],Iy[i,j,1]]]) for i in range(height) for j in range(width)]

    det,trace = list(map(np.linalg.det,IxIy)),list(map(np.trace,IxIy))
    R = np.array([det - 0.04 * trace * trace for det,trace in zip(det,trace)])
    R_max = np.max(R)
    R = R.reshape(height,width)
    corner = np.zeros_like(R,dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if R[i, j] > R_max * 0.1 and R[i, j] == np.max(R[max(0, i - 1):min(i + 2, height - 1), max(0, j - 1):min(j + 2, width - 1)]):
                corner[i, j] = 255

    orientation_image = np.zeros(img.shape[:2])
    orientation_image = np.degrees(np.arctan2(Ix[:,:,1].flatten(), Ix[:,:, 0].flatten()).reshape(orientation_image.shape))

    return R, corner, orientation_image


def scale_image(image,ksizes):
    height,width = image.shape[:2]
    results = []

    for ksize in ksizes:
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        image = cv2.Laplacian(image, cv2.CV_16S, ksize)

        sobel = np.zeros((height, width, 2), dtype=np.float32)
        sobel[:,:,0] = cv2.Sobel(image,cv2.CV_16S,1,0,ksize)
        sobel[:,:,1] = cv2.Sobel(image,cv2.CV_16S,0,1,ksize)

        Ix = np.zeros((height,width,3),dtype=np.float32)
        Ix[:,:,0] = sobel[:,:,0]*sobel[:,:,0]
        Ix[:,:,1] = sobel[:,:,1]*sobel[:,:,1]
        Ix[:,:,2] = sobel[:,:,0]*sobel[:,:,1]

        Ix = [np.array([[Ix[i,j,0],Ix[i,j,2]],[Ix[i,j,2],Ix[i,j,1]]]) for i in range(height) for j in range(width)]
        det, trace = list(map(np.linalg.det,Ix)),list(map(np.trace,Ix))

        R = np.array([d-0.04*t**2 for d,t in zip(det,trace)])
        R = R.reshape(height, width)
        results.append(R)

    temp = np.array([result for result in results])
    result_max = np.argmax(temp, axis=0)
    return result_max

def detect_keypoints(image):
    height, width = image.shape[:2]
    features = []
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harrisImage, harrisMaxImage, orientationImage = harris_corner_detector(grayImage)
    scale = scale_image(grayImage, range(1, 20, 2))

    for y in range(height):
        for x in range(width):
            if harrisMaxImage[y, x] != 0:
                kp = cv2.KeyPoint()
                kp.size = int(scale[y, x])
                kp.pt = (x, y)
                kp.angle = orientationImage[y, x]
                kp.response = harrisImage[y, x]
                features.append(kp)
    return features

def describeFeatures(image, keypoints):
    image = image.astype(np.float32)
    image /= 255.
    windowSize = 8
    desc = np.zeros((len(keypoints), windowSize * windowSize))
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = ndimage.gaussian_filter(grayImage, 0.5)

    for i, f in enumerate(keypoints):
        x0, y0 = f.pt
        theta = - f.angle / 180.0 * np.pi
        T1 = np.array([[1, 0, -x0],[0, 1, -y0],[0, 0, 1]])
        T2 = np.array([[1, 0, 4], [0, 1, 4], [0, 0, 1]])
        R = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        S = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]])
        MF = np.dot(np.dot(np.dot(T2, S), R), T1)
        transMx = MF[0:2,0:3]
        destImage = cv2.warpAffine(grayImage, transMx,(windowSize, windowSize), flags=cv2.INTER_LINEAR)
        target = destImage[:8, :8]
        target = target - np.mean(target)
        if np.std(target) <= 10**(-5):
            desc[i, :] = np.zeros((windowSize * windowSize,))
        else:
            target = target / np.std(target)
            desc[i,:] = target.reshape(windowSize * windowSize)

    return desc


def match_features(desc1, desc2):
    matches = []
    assert desc1.ndim == 2
    assert desc2.ndim == 2
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    for i in range(desc1.shape[0]):
        diff = (desc2 - desc1[i])*(desc2 - desc1[i])
        sum_diff = diff.sum(axis = 1)
        dis = sum_diff ** 0.5
        argmin = np.argmin(dis)
        match = cv2.DMatch()
        match.queryIdx = i
        match.trainIdx = int(argmin)
        match.distance = dis[argmin]
        matches.append(match)

    return matches


def generate_harris_corner(image_1, image_2):
    keypoint_1 = detect_keypoints(image_1)
    keypoint_2 = detect_keypoints(image_2)

    des1 = describeFeatures(image_1, keypoint_1)
    des2 = describeFeatures(image_2, keypoint_2)

    cv2.drawKeypoints(image=image_1, outImage=image_1, keypoints=keypoint_1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 0, 255))
    cv2.drawKeypoints(image=image_2, outImage=image_2, keypoints=keypoint_2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 0, 255))

    measure_match = match_features(des1, des2)
    output = cv2.drawMatches(image_1, keypoint_1, image_2, keypoint_2, measure_match[:50], None, flags=2)

    return output


if __name__=='__main__':
    #image = cv2.imread('data/panda1.jpg')
    # rotated_image = ndimage.rotate(image, 68)
    # translation_image = ndimage.shift(image, np.array([130, 290, 0]))
    #scale_image = cv2.resize(image, None, fx=30, fy=30, interpolation=cv2.INTER_CUBIC)

    # cv2.imwrite("results/image.jpg", image)
    #cv2.imwrite("results/image_rotated.jpg", rotated_image)
    # cv2.imwrite("results/image_translation.jpg", translation_image)
    #cv2.imwrite("results/image_scale.jpg", scale_image)
    image_1 = cv2.imread('results/image.jpg')
    image_2 = cv2.imread('results/image_rotated.jpg')
    image_3 = cv2.imread('results/image_translation.jpg')
    image_4 = cv2.imread('results/image_scale.jpg')

    rotated_output = generate_harris_corner(image_1,image_2)
    cv2.imwrite("results/harris_rotated.jpg", rotated_output)

    translate_output = generate_harris_corner(image_1, image_3)
    cv2.imwrite("results/harris_translated.jpg", translate_output)

    scale_output = generate_harris_corner(image_4, image_1)
    cv2.imwrite("results/harris_scale.jpg", scale_output)
