import numpy as np
import os
import cv2
from PIL import Image
import pickle
import global_variables

base_dir = 'dataset'

face_cascade = global_variables.face_cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        # save the path
        path = os.path.join(root, file)

        # create the correspondent label
        label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
        if not label in label_ids:
            label_ids[label] = current_id
            current_id+=1
        id_ = label_ids[label]

        pil_image = Image.open(path).convert("L") # open the image and convert into grayscale one
        image_array = np.array(pil_image, "uint8") # save the images into numpy array
        # detection func -> look into global_variables.py files to check scaleFacetor and minNeighbors value
        faces = face_cascade.detectMultiScale(image_array, global_variables.scaleFactor, global_variables.minNeighbors)
        # 1. bring the face 2. append the train and the labels
        for (x, y, w, h) in faces:
            roi = image_array[y:y+h, x:x+w]
            x_train.append(roi)
            y_labels.append(id_)

# save labels in pickel file
with open("trainer/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# trains the recognizer with given data (training images, that means the faces you want to learn) and associated labels corresponding to the images
recognizer.train(x_train, np.array(y_labels))
# save in the trainer file
recognizer.save("trainer/trainer.yml")