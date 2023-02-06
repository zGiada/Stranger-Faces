# this is the code used for fix the dataset image
import uuid
import os
import cv2
import global_variables

face_cascade = global_variables.face_cascade

Source_Path = 'code-dataset/base/'
destination = 'code-dataset/fix/'
files = os.listdir(Source_Path)
count=0
for index, file in enumerate(files):
    image = cv2.imread(Source_Path + file)
    faces = face_cascade.detectMultiScale(image, global_variables.scaleFactor, global_variables.minNeighbors)

    for (x, y, w, h) in faces:
        rectangle = image[y:y + h, x:x + w]
        new_img = destination + str(uuid.uuid4()).replace("-", "") + '.jpg'
        # check a min size
        if (rectangle.shape[0] > 80 and rectangle.shape[1] > 80) or ((
                rectangle.shape[0] > 80 and rectangle.shape[1] <= 80) or (
                    rectangle.shape[1] > 80 and rectangle.shape[0] <= 80)):
            cv2.imwrite(new_img, cv2.resize(rectangle, (500,500), interpolation = cv2.INTER_AREA))
        else:
            count += 1
            print(Source_Path +""+file)