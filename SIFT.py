import cv2

image_1 = cv2.imread('results/image.jpg')
image_2 = cv2.imread('results/image_scale.jpg')

SIFT = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = SIFT.detectAndCompute(image_1,None)
keypoints_2, descriptors_2 = SIFT.detectAndCompute(image_2,None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
output = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches[:50], image_2, flags=2)
cv2.imwrite("results/SIFT_scale.jpg", output)