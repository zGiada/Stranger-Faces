import cv2
import statistics
import pickle
import global_variables

face_cascade = global_variables.face_cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

labels = {}
with open("trainer/labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture('starting_media/interview.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening file")

# ----- variables for the evaluation
# conf_values_name -> save all the confidence value, in order to do a final mean
conf_values_fin = []
conf_values_mil = []
conf_values_noah = []
# count_noah_frames, count_millie_frames, count_finn_frames -> count the frames for each actors
# count -> count the total frames
# unknown -> count the times of "unknown" result
count_noah_frames, count_millie_frames, count_finn_frames, count, unknown = 0, 0, 0, 0, 0

# Read until video is completed
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        count += 1 # count the frames
        gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert into grayscale
        # detection func -> look into global_variables.py files to check scaleFacetor and minNeighbors value
        faces = face_cascade.detectMultiScale(gray, global_variables.scaleFactor, global_variables.minNeighbors, minSize=(200, 200))
        color = (0, 0, 255)
        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]
            id_, conf = recognizer.predict(face_gray)
            if conf >= 55: # quite low value, depends on dataset dimensions and quality of video (check report details)
                if ("noah" in (labels[id_])):
                    count_noah_frames += 1
                    conf_values_noah.append(round(conf))
                if ("millie" in (labels[id_])):
                    count_millie_frames += 1
                    conf_values_mil.append(round(conf))
                if ("finn" in (labels[id_])):
                    count_finn_frames += 1
                    conf_values_fin.append(round(conf))
                tex = (labels[id_]).replace("-"," ") + " (" + str(round(conf)) + "%)"
                cv2.putText(frame, tex, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "unknown "+ " (" + str(round(conf)) + "%)", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                unknown += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display the resulting frame
        cv2.imshow('Stranger Things happens...', cv2.resize(frame, (960, 540)))
        
        if cv2.waitKey(8) == ord('q'):
            break    
    else:
        break

# release the video capture object
cap.release()

# closes window
cv2.destroyAllWindows()


# evaluation result
# for each actors (and for the unknown) -> counted_frame/count - value% - mean conf
print(":: resume ::\n\nFRAMED:\n"
      + labels[0] + " \t\t= " + str(count_finn_frames) + "/"+str(count)+" ("+(str(round(count_finn_frames/count*100, 2)))+"%"+" of frame)\twith a mean conf of "+str(round(statistics.mean(conf_values_fin)))+"% \n"
      + labels[1] + " \t= " + str(count_millie_frames) + "/"+str(count)+" ("+(str(round(count_millie_frames/count*100, 2)))+"%"+" of frame)\twith a mean conf of "+str(round(statistics.mean(conf_values_mil)))+"% \n"
      + labels[2] + " \t\t= " + str(count_noah_frames) + "/"+str(count)+" ("+(str(round(count_noah_frames/count*100, 2)))+"%"+" of frame)\twith a mean conf of "+str(round(statistics.mean(conf_values_noah)))+"% \n\n"
      + "unknown = "+ str(unknown)+ "/"+str(count)+" ("+(str(round(unknown/count*100, 2)))+"%)\n\n")
