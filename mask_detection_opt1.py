#import libraries
import os
import numpy
import cv2
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from PIL import Image
from keras.models import load_model
import cv2
import numpy as np
import time
mo = load_model('mask_classidication_opt1_model.h5')
#mo.compile(loss='binary_crossentropy',
               #optimizer='rmsprop',
              # metrics=['accuracy'])
#get the absolute path of the working directory
dir_path = os.path.dirname(os.path.realpath(__file__))


#Reads the network model stored in Caffe framework's format.
model = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'weights.caffemodel')
print("press 1: for uploading image.")
print("press 2: for uploading live")
n=int(input("press:"))
if(n==1):
  # split the file name and the extension into two variales
  filename = "test_00000642.jpg"
  # check if the file extension is .png,.jpeg or .jpg
  # read the image using cv2
  image = cv2.imread(filename)
  # accessing the image.shape tuple and taking the elements
  (h, w) = image.shape[:2]

  # get our blob which is our input image
  blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                               (103.93, 116.77, 123.68))  # input the blob into the model and get back the detections
  model.setInput(blob)
  detections = model.forward()
  # Iterate over all of the faces detected and extract their start and end points

  for i in range(0, detections.shape[2]):
    box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    confidence = detections[0, 0, i, 2]
    # if the algorithm is more than 16.5% confident that the      detection is a face, show a rectangle around it
    if (confidence > 0.2):
      #cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
      try:
        iii = Image.open(filename)
        crop_image = iii.crop((startX, startY, endX, endY))
        new_img = crop_image.resize((150,150))
        m= load_model('mask_classidication_opt1_model.h5')
        m.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        new_img= np.reshape(new_img, [1, 150,150, 3])
        classes = m.predict_classes(new_img)
        i = int(classes)
        if i == 0:
          cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
          s="person has mask"
          print("person has mask")
          text = "{:.2f}%".format(confidence * 100)
          cv2.putText(image, text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
          cv2.putText(image, s, (startX, endY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        elif i == 1:
          cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
          s="person has no mask"
          print("person has no mask")
          text = "{:.2f}%".format(confidence * 100)
          cv2.putText(image, text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
          cv2.putText(image, s, (startX, endY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2)

      except:
        print("unable to crop")
  imshow('face detection', image)
  if cv2.getWindowProperty('face detection', cv2.WND_PROP_VISIBLE) == -1:
    #waitKey(0)
    # close the window

    cv2.destroyAllWindows()
  else:
    waitKey(0)
    # close the window
    cv2.destroyAllWindows()
  cv2.destroyAllWindows()


elif(n==2):
  cap = cv2.VideoCapture(0)
  cap.set(3, 640)  # set Width
  cap.set(4, 480)  # set Height
  while True:
    now=time.time()
    ret, image = cap.read()
    # img = cv2.flip(img, -1)

    (h, w) = image.shape[:2]

    # get our blob which is our input image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                 (103.93, 116.77, 123.68))  # input the blob into the model and get back the detections
    model.setInput(blob)
    detections = model.forward()


    for i in range(0, detections.shape[2]):
      box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      confidence = detections[0, 0, i, 2]
      # if the algorithm is more than 16.5% confident that the      detection is a face, show a rectangle around it
      if (confidence > 0.2):
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
        try:
          face=image[startY:endY,startX:endX]

          new_img = cv2.resize(face,(150,150))
          #print("went")
          #mo = load_model('mask_classidication_opt1_model.h5')
          #mo.compile(loss='binary_crossentropy',
                        #optimizer='rmsprop',
                       # metrics=['accuracy'])
          new_img = np.reshape(new_img, [1, 150,150, 3])
          classes = mo.predict_classes(new_img)
          end=time.time()
          f=1//(end-now)
          fps='fps: '+str(f)
          i = int(classes)
          if i == 0:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            s = "person has mask"
            print("person has mask")
            text = "{:.2f}% ".format(confidence * 100)
            text=fps+'--'+text
            cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(image, s, (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

          elif i == 1:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            s = "person has no mask"
            print("person has no mask")
            text = "{:.2f}%".format(confidence * 100)
            text = fps + '--' + text
            cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, s, (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        except:
          print("unable to crop")


        # save the modified image to the Output folder
        #text = "{:.2f}%".format(confidence * 100)
        #cv2.putText(image, text, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    cv2.imshow('video', image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
      break
    if cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1:
      break
  cap.release()
  cv2.destroyAllWindows()
else:
  print("enter correct option")

