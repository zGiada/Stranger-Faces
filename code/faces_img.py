import random
import cv2
import pickle
import global_variables

def face_detect_recogn(image_read, path_write):
    image = cv2.imread(image_read)

    face_cascade = global_variables.face_cascade

    # open trained and labels file for recognition
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")
    labels = {}
    with open("trainer/labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert into a gray image
    # detection func -> look into global_variables.py files to check scaleFacetor and minNeighbors value
    faces = face_cascade.detectMultiScale(gray, global_variables.scaleFactor, global_variables.minNeighbors, minSize=(100,100))

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        # Predicts a label and associated confidence for a given input image
        id_, conf = recognizer.predict(face_gray)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # usually I set that on 70, but..
        if conf >= 45:
            # ..now it is a low value, only to demonstrate the fact that in the serie they looks different
            # (and dataset must be increased) and only to show value graphically on the resulted image
            tex = (labels[id_]).replace("-", " ") + " (" + str(round(conf)) + "%)"
            cv2.putText(image, tex, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            # if the tool are not capable to recognize, put the "unknown" text
            tex = "unknown ("+str(round(conf))+"%)"
            cv2.putText(image, tex, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # drow the rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    cv2.imwrite(path_write, image)

# from interview image
face_detect_recogn("starting_media/interview_img.jpg", "resulted_images/interview_recognition.png")
# from TV serie image
face_detect_recogn("starting_media/serie_img.jpg", "resulted_images/season4_recognition.png")
