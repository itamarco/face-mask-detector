import cv2
import numpy as np
from keras.models import load_model

TARGET_HEIGHT = 140
TARGET_WIDTH = 140
RESIZE_FACTOR = 4

model=load_model("./assets/face-mask-model.h5")

labels_dict={0:'no mask',1:'with mask'}
color_dict={0:(0,0,255),1:(0,255,0)}


webcam = cv2.VideoCapture(0)

# OpenCV front face classifier
classifier = cv2.CascadeClassifier('./assets/haarcascade_frontalface_default.xml')

while True:
    (rval, image_capture) = webcam.read()
    image_capture=cv2.flip(image_capture,1,1) #Flip to act as a mirror

    # Minify the image to speed up face detection
    minified_img_capture = cv2.resize(image_capture, (image_capture.shape[1] // RESIZE_FACTOR, image_capture.shape[0] // RESIZE_FACTOR))

    # detect MultiScale / faces 
    gray = cv2.cvtColor(minified_img_capture, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(60, 60)
    )

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * RESIZE_FACTOR for v in f] #Scale back the face coordinates
        
        face_img = image_capture[y:y+h, x:x+w]

        # Prepare image for detection
        resized=cv2.resize(face_img,(TARGET_HEIGHT,TARGET_WIDTH))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,TARGET_HEIGHT,TARGET_WIDTH,3))
        reshaped = np.vstack([reshaped])

        result=model.predict(reshaped)
      
        # prepare detection info info
        certainity=np.amax(result, axis=1)[0]
        label=np.argmax(result,axis=1)[0]
        label_value = labels_dict[label]
        caption= "{0} {1:.3f}".format(label_value, certainity)
        color=color_dict[label]
      
        # draw detection info on image
        cv2.rectangle(image_capture,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(image_capture,(x,y-40),(x+w,y),color,-1)
        cv2.putText(image_capture, caption, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # Show live detection
    cv2.imshow('LIVE',   image_capture)
    key = cv2.waitKey(10) # wait for user key
    if key == 27: #Esc key
        break

webcam.release()

cv2.destroyAllWindows()