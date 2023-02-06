import cv2
# variables that are used in more than one tools
scaleFactor = 1.2
minNeighbors = 5
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')