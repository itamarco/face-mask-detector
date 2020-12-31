import cv2
import numpy as np
from keras.models import load_model

TARGET_HEIGHT = 140
TARGET_WIDTH = 140

def showImage(img): 
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    if k == 27: # wait for ESC key to exit
        cv2.destroyAllWindows()


model=load_model("./assets/face-mask-model.h5")

labels_dict={0:'no mask',1:'with mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

image = cv2.imread('./assets/face-masks.jpg')

classifier = cv2.CascadeClassifier('./assets/haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags=cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    face_img = image[y:y+h, x:x+w]

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
    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
    cv2.rectangle(image,(x,y-40),(x+w,y),color,-1)
    cv2.putText(image, caption, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

showImage(image)
cv2.destroyAllWindows()

